# 라이브러리 불러오기
import numpy as np
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# 파라미터 값 세팅
state_size = [3*2, 64, 64]
action_size = 5

load_model = False
train_mode = True

discount_factor = 0.999
learning_rate = 3e-4
n_step = 32
batch_size = 512
n_epoch = 3
_lambda = 0.95
epsilon = 0.2

rnd_learning_rate = 1e-4
rnd_strength = 0.1
rnd_discount_factor = 0.99
rnd_feature_size = 128

run_step = 5000000 if train_mode else 10000

print_interval = 1
save_interval = 100

# 유니티 환경 경로
game = "Tetris"
env_name = f"./Env/{game}"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H")
save_path = f"./saved_models/{date_time}"
load_path = f"./saved_models/2024050217"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device set as {device}')


# ActorCritic 클래스 -> Actor Network, Critic Network 정의
class PPONetwork(torch.nn.Module):
    def __init__(self, **kwargs):
        super(PPONetwork, self).__init__(**kwargs)
        self.conv1_kernel_size = 4
        self.conv1_stride = 2
        self.conv1_out_channel = 16
        self.conv1 = torch.nn.Conv2d(in_channels=state_size[0], out_channels=self.conv1_out_channel,
                                     kernel_size=self.conv1_kernel_size, stride=self.conv1_stride)
        dim1 = ((state_size[1] - self.conv1_kernel_size) // self.conv1_stride + 1, (state_size[2] - self.conv1_kernel_size) // self.conv1_stride + 1)

        self.conv2_kernel_size = 4
        self.conv2_stride = 2
        self.conv2_out_channel = 32
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1_out_channel, out_channels=self.conv2_out_channel,
                                     kernel_size=self.conv2_kernel_size, stride=self.conv2_stride)
        dim2 = ((dim1[0] - self.conv2_kernel_size) // self.conv2_stride + 1, (dim1[1] - self.conv2_kernel_size) // self.conv2_stride + 1)
        self.flat = torch.nn.Flatten()

        self.a_d1 = torch.nn.Linear(32 * dim2[0] * dim2[1], 512)
        self.a_d2 = torch.nn.Linear(512, 512)
        self.a_d3 = torch.nn.Linear(512, 512)
        self.pi = torch.nn.Linear(512, action_size)

        self.c_d1 = torch.nn.Linear(32 * dim2[0] * dim2[1], 512)
        self.c_d2 = torch.nn.Linear(512, 512)
        self.c_d3 = torch.nn.Linear(512, 512)
        self.v = torch.nn.Linear(512, 1)

        self.v_d1 = torch.nn.Linear(32 * dim2[0] * dim2[1], 512)
        self.v_d2 = torch.nn.Linear(512, 512)
        self.v_d3 = torch.nn.Linear(512, 512)
        self.v_i = torch.nn.Linear(512, 1)

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        x = x.permute(0, 1, 2, 3)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.flat(x)

        ax = F.leaky_relu(self.a_d1(x))
        ax = F.leaky_relu(self.a_d2(ax))
        ax = F.leaky_relu(self.a_d3(ax))

        cx = F.leaky_relu(self.c_d1(x))
        cx = F.leaky_relu(self.c_d2(cx))
        cx = F.leaky_relu(self.c_d3(cx))
        return F.softmax(self.pi(ax), dim=-1), self.v(cx)

    def get_vi(self, x):
        x = x.permute(0, 1, 2, 3)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.flat(x)
        x = F.leaky_relu(self.v_d1(x))
        x = F.leaky_relu(self.v_d2(x))
        x = F.leaky_relu(self.v_d3(x))
        return self.v_i(x)


class RNDNetwork(torch.nn.Module):
    def __init__(self, is_predictor, **kwargs):
        super(RNDNetwork, self).__init__(**kwargs)
        self.is_predictor = is_predictor
        self.conv1 = torch.nn.Conv2d(in_channels=state_size[0], out_channels=16,
                                     kernel_size=8, stride=4)
        dim1 = ((state_size[1] - 8) // 4 + 1, (state_size[2] - 8) // 4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32,
                                     kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4) // 2 + 1, (dim1[1] - 4) // 2 + 1)
        self.flat = torch.nn.Flatten()
        if is_predictor:
            self.d1 = torch.nn.Linear(32 * dim2[0] * dim2[1], 128)
            self.d2 = torch.nn.Linear(128, 128)
            self.feature = torch.nn.Linear(128, rnd_feature_size)
        else:
            self.feature = torch.nn.Linear(32 * dim2[0] * dim2[1], rnd_feature_size)

    def forward(self, x):
        x = x.permute(0, 1, 2, 3)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.flat(x)
        if self.is_predictor:
            x = F.relu(self.d1(x))
            x = F.relu(self.d2(x))
        return self.feature(x)


class RunningMeanStd(torch.nn.Module):
    def __init__(self, shape, epsilon=1e-4):
        super(RunningMeanStd, self).__init__()
        self.mean = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.var = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.count = torch.nn.Parameter(torch.tensor(epsilon), requires_grad=False)

    def update(self, x):
        batch_mean, batch_std, batch_count = x.mean(axis=0), x.std(axis=0), x.shape[0]
        batch_var = torch.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = (
                m_a
                + m_b
                + torch.square(delta)
                * self.count
                * batch_count
                / (self.count + batch_count)
        )
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean.data = new_mean
        self.var.data = new_var
        self.count.data = new_count

    def normalize(self, x):
        return torch.clip((x - self.mean) / (torch.sqrt(self.var) + 1e-7), min=-5.0, max=5.0)


# PPOAgent 클래스 -> PPO 알고리즘을 위한 다양한 함수 정의
class Agent:
    def __init__(self, id):
        self.random_network = RNDNetwork(is_predictor=False).to(device)
        self.predictor_network = RNDNetwork(is_predictor=True).to(device)
        self.rnd_optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=rnd_learning_rate)

        # 관측, 내적 보상 RunningMeanStd 선언
        raw_state_size = state_size # CWH -> WHC
        self.obs_rms = RunningMeanStd(raw_state_size).to(device)
        self.ri_rms = RunningMeanStd(1).to(device)

        self.save_path = f"{save_path}/{id}"
        self.load_path = f"{load_path}/{id}"

        self.network = PPONetwork().to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = list()
        self.writer = SummaryWriter(self.save_path)

        if load_model:
            print(f"... Load Model from {self.load_path} ...")
            checkpoint = torch.load(self.load_path + '/ckpt', map_location=device)
            self.network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    # 정책을 통해 행동 결정
    def get_action(self, state, training=True):
        # 네트워크 모드 설정
        self.network.train(training)

        # 네트워크 연산에 따라 행동 결정
        pi, _ = self.network(torch.FloatTensor(state).to(device))
        action = torch.multinomial(pi, num_samples=1).cpu().numpy()
        return action

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 학습 수행
    def train_model(self):
        self.network.train()
        self.predictor_network.train()
        self.random_network.train(False)

        state = np.stack([m[0] for m in self.memory], axis=0)
        action = np.stack([m[1] for m in self.memory], axis=0)
        reward = np.stack([m[2] for m in self.memory], axis=0)
        next_state = np.stack([m[3] for m in self.memory], axis=0)
        done = np.stack([m[4] for m in self.memory], axis=0)
        self.memory.clear()

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                      [state, action, reward, next_state, done])

        # obs_rms update
        self.obs_rms.update(state)

        # prob_old, adv, ret 계산
        with torch.no_grad():
            # obs normalize
            normalized_next_state = self.obs_rms.normalize(next_state)
            target = self.random_network(normalized_next_state)
            pred = self.predictor_network(normalized_next_state)
            reward_i = torch.sum(torch.square(pred - target), dim=1, keepdim=True)

            # ri_rms update
            self.ri_rms.update(reward_i)

            # ri normalize
            reward_i /= torch.sqrt(self.ri_rms.var) + 1e-7

            pi_old, value = self.network(state)
            prob_old = pi_old.gather(1, action.long())
            value_i = self.network.get_vi(state)

            _, next_value = self.network(next_state)
            delta = reward + (1 - done) * discount_factor * next_value - value

            next_value_i = self.network.get_vi(next_state)
            delta_i = reward_i + rnd_discount_factor * next_value_i - value_i

            adv, adv_i = delta.clone(), delta_i.clone()
            adv, adv_i, done = map(lambda x: x.view(n_step, -1).transpose(0, 1).contiguous(), [adv, adv_i, done])
            for t in reversed(range(n_step - 1)):
                adv[:, t] += (1 - done[:, t]) * discount_factor * _lambda * adv[:, t + 1]
                # non episodic for internal advantage
                adv_i[:, t] += rnd_discount_factor * _lambda * adv_i[:, t + 1]

            # adv normalization
            adv = (adv - adv.mean(dim=1, keepdim=True)) / (adv.std(dim=1, keepdim=True) + 1e-7)
            adv_i = (adv_i - adv_i.mean(dim=1, keepdim=True)) / (adv_i.std(dim=1, keepdim=True) + 1e-7)

            adv, adv_i = map(lambda x: x.transpose(0, 1).contiguous().view(-1, 1), [adv, adv_i])

            ret = adv + value
            ret_i = adv_i + value_i

            # merge internal and external
            adv = adv + rnd_strength * adv_i

        # 학습 이터레이션 시작
        actor_losses, critic_losses, rnd_losses = [], [], []
        idxs = np.arange(len(reward))
        for _ in range(n_epoch):
            np.random.shuffle(idxs)
            for offset in range(0, len(reward), batch_size):
                idx = idxs[offset: offset + batch_size]

                _state, _next_state, _action, _ret, _ret_i, _adv, _prob_old = \
                    map(lambda x: x[idx], [state, next_state, action, ret, ret_i, adv, prob_old])

                pi, value = self.network(_state)
                prob = pi.gather(1, _action.long())

                # 정책신경망 손실함수 계산
                ratio = prob / (_prob_old + 1e-7)
                surr1 = ratio * _adv
                surr2 = torch.clamp(ratio, min=1 - epsilon, max=1 + epsilon) * _adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # 가치신경망 손실함수 계산
                critic_loss_e = F.mse_loss(value, _ret).mean()

                # 내적 가치 신경망 손실함수 계산
                value_i = self.network.get_vi(_state)
                critic_loss_i = F.mse_loss(value_i, _ret_i).mean()

                critic_loss = critic_loss_e + critic_loss_i

                total_loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # RND 신경망 손실함수 계산
                _normalized_next_state = self.obs_rms.normalize(_next_state)
                with torch.no_grad():
                    target = self.random_network(_normalized_next_state)
                pred = self.predictor_network(_normalized_next_state)
                rnd_loss = torch.sum(torch.square(pred - target), dim=1).mean()

                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self.rnd_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                rnd_losses.append(rnd_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses), np.mean(rnd_losses)

    # 네트워크 모델 저장
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, save_path + '/ckpt')

    # 학습 기록
    def write_summary(self, score, actor_loss, critic_loss, rnd_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)
        self.writer.add_scalar("model/rnd_loss", rnd_loss, step)


# Main 함수 -> 전체적으로 Adversarial PPO 알고리즘을 진행 
if __name__ == '__main__':
    # 유니티 환경 경로 설정 (file_name)
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                           side_channels=[engine_configuration_channel])
    env.reset()

    # 유니티 behavior 설정 
    behavior_name_list = list(env.behavior_specs.keys())
    behavior_A = behavior_name_list[0]
    behavior_B = behavior_name_list[1]
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
    dec_A, term_A = env.get_steps(behavior_A)
    dec_B, term_B = env.get_steps(behavior_B)

    # PPOAgent 클래스를 agent_A, agent_B로 정의 
    agent_A = Agent("A")
    agent_B = Agent("B")

    episode = 0
    actor_losses_A, critic_losses_A, rnd_losses_A, score_A = [], [], [], 0
    actor_losses_B, critic_losses_B, rnd_losses_B, score_B = [], [], [], 0
    for step in range(run_step):
        if step == run_step:
            if train_mode:
                agent_A.save_model()
                agent_B.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

        state_A = dec_A.obs[0]
        action_A = agent_A.get_action(state_A, train_mode)

        state_B = dec_B.obs[0]
        action_B = agent_B.get_action(state_B, train_mode)
        action_tuple_A, action_tuple_B = map(lambda x: ActionTuple(discrete=x), [action_A, action_B])

        env.set_actions(behavior_A, action_tuple_A)
        env.set_actions(behavior_B, action_tuple_B)
        env.step()

        dec_A, term_A = env.get_steps(behavior_A)
        done_A = len(term_A.agent_id) > 0
        next_state_A = term_A.obs[0] if done_A else dec_A.obs[0]
        reward_A = term_A.reward if done_A else dec_A.reward
        score_A += reward_A[0]

        dec_B, term_B = env.get_steps(behavior_B)
        done_B = len(term_B.agent_id) > 0
        next_state_B = term_B.obs[0] if done_B else dec_B.obs[0]
        reward_B = term_B.reward if done_B else dec_B.reward
        score_B += reward_B[0]

        if train_mode:
            agent_A.append_sample(state_A[0], action_A[0], reward_A, next_state_A[0], [done_A])
            agent_B.append_sample(state_B[0], action_B[0], reward_B, next_state_B[0], [done_B])

            if (step + 1) % n_step == 0:
                # 학습 수행
                actor_loss_A, critic_loss_A, rnd_loss_A = agent_A.train_model()
                actor_losses_A.append(actor_loss_A)
                critic_losses_A.append(critic_loss_A)
                rnd_losses_A.append(rnd_loss_A)

                actor_loss_B, critic_loss_B, rnd_loss_B = agent_B.train_model()
                actor_losses_B.append(actor_loss_B)
                critic_losses_B.append(critic_loss_B)
                rnd_losses_B.append(rnd_loss_A)

        if done_A or done_B:
            episode += 1

            mean_rnd_loss_A = np.mean(rnd_losses_A) if len(rnd_losses_A) > 0 else 0
            mean_actor_loss_A = np.mean(actor_losses_A) if len(actor_losses_A) > 0 else 0
            mean_critic_loss_A = np.mean(critic_losses_A) if len(critic_losses_A) > 0 else 0
            agent_A.write_summary(score_A, mean_actor_loss_A, mean_critic_loss_A, mean_rnd_loss_A, step)

            mean_rnd_loss_B = np.mean(rnd_losses_B) if len(rnd_losses_B) > 0 else 0
            mean_actor_loss_B = np.mean(actor_losses_B) if len(actor_losses_B) > 0 else 0
            mean_critic_loss_B = np.mean(critic_losses_B) if len(critic_losses_B) > 0 else 0
            agent_B.write_summary(score_B, mean_actor_loss_B, mean_critic_loss_B, mean_rnd_loss_B, step)

            print(f"{episode} Episode / Step: {step} / " + \
                  f"A Score: {score_A:.5f} / B Score: {score_B:.5f} / " + \
                  f"A Actor Loss: {mean_actor_loss_A:.4f} / B Actor Loss: {mean_actor_loss_B:.4f} / " + \
                  f"A Critic Loss: {mean_critic_loss_A:.4f} / B Critic Loss: {mean_critic_loss_B:.4f} / " + \
                  f"A rnd Loss: {mean_rnd_loss_A:.4f} / B rnd Loss: {mean_rnd_loss_B:.4f}")

            score_A = score_B = 0
            actor_losses_A, critic_losses_A, rnd_losses_A = [], [], []
            actor_losses_B, critic_losses_B, rnd_losses_B = [], [], []

            # 네트워크 모델 저장 
            if train_mode and episode % save_interval == 0:
                agent_A.save_model()
                agent_B.save_model()

    agent_A.save_model()
    agent_B.save_model()

    env.close()
