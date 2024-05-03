# 라이브러리 불러오기
import copy

import numpy as np
import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import random
from collections import deque

# 파라미터 값 세팅
state_size = [1*3, 64, 64]
action_size = 5

load_model = False
train_mode = True

rnd_learning_rate = 1e-5
rnd_feature_size = 128

run_step = 5000000 if train_mode else 10000

train_interval = 32
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


class DQNNetwork(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DQNNetwork, self).__init__(**kwargs)
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

        self.fc1 = torch.nn.Linear(32 * dim2[0] * dim2[1], 1024)
        # self.fc1 = torch.nn.Linear(3 * 64 * 64, 4096)
        self.fc2 = torch.nn.Linear(1024, 2048)
        self.fc3 = torch.nn.Linear(2048, 2048)
        self.fc4 = torch.nn.Linear(2048, 1024)
        self.fc5 = torch.nn.Linear(1024, 512)
        self.fc6 = torch.nn.Linear(512, 256)
        self.fc_out = torch.nn.Linear(256, action_size)

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        x = x.permute(0, 1, 2, 3)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.flat(x)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = F.leaky_relu(self.fc6(x))

        return F.softmax(self.fc_out(x), dim=-1)


class NN(torch.nn.Module):
    def __init__(self, is_predictor):
        super(NN, self).__init__()
        self.is_predictor = is_predictor

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

        if self.is_predictor:
            self.d1 = torch.nn.Linear(32 * dim2[0] * dim2[1], 1024)
            # self.d1 = torch.nn.Linear(3 * 64 * 64, 2048)
            self.d2 = torch.nn.Linear(1024, 1024)
            self.d3 = torch.nn.Linear(1024, 1024)
            self.d4 = torch.nn.Linear(1024, 512)
            self.d5 = torch.nn.Linear(512, 256)
            self.d6 = torch.nn.Linear(256, 256)
            self.feature = torch.nn.Linear(256, rnd_feature_size)
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
            x = F.relu(self.d3(x))
            x = F.relu(self.d4(x))
            x = F.relu(self.d5(x))
            x = F.relu(self.d6(x))
        return self.feature(x)


class RNDNetwork(torch.nn.Module):
    def __init__(self):
        super(RNDNetwork, self).__init__()
        self.random_network = NN(is_predictor=False).to(device)
        self.predictor_network = NN(is_predictor=True).to(device)
        self.optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=rnd_learning_rate)

    def get_reward(self, x):
        y_true = self.random_network(x).detach()
        y_pred = self.predictor_network(x)
        reward = torch.pow(y_pred - y_true, 2).sum()
        return reward

    def update(self, reward):
        reward.sum().backward()
        self.optimizer.step()


# PPOAgent 클래스 -> PPO 알고리즘을 위한 다양한 함수 정의
class Agent:
    def __init__(self, id):
        self.learning_rate = 3*1e-7
        self.memory_size = 512
        self.batch_size = 64
        self._lambda = 0.95

        self.save_path = f"{save_path}/{id}"
        self.load_path = f"{load_path}/{id}"

        self.network = DQNNetwork().to(device)
        self.target_network = copy.deepcopy(self.network).to(device)
        self.rnd = RNDNetwork()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        self.epsilon = 1.0
        self.epsilon_decay = 1 / (run_step + 1e-7)
        self.epsilon_min = 0.1

        self.memory = deque(maxlen=self.memory_size)
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
        q = self.network(torch.FloatTensor(state).to(device))
        if np.random.rand() <= self.epsilon:
            if np.random.rand() >= 0.5:
                action = torch.multinomial(q, num_samples=1).cpu().numpy()
            else:
                action = np.random.randint(0, action_size, size=(1,1))

        else:
            action = [torch.argmax(q, dim=1).to("cpu").tolist()]
            action = np.array(action)

        return action

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    # 학습 수행
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        self.network.train()

        samples_size = np.min([len(self.memory), self.batch_size])
        samples = random.sample(self.memory, samples_size)

        states, actions, rewards, next_states, dones = zip(*samples)
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long).view(samples_size, -1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(samples_size, -1).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        rnd_loss = self.rnd.get_reward(states)
        adjusted_rewards = rewards + rnd_loss.clone().detach()
        self.rnd.update(rnd_loss)
        target_q = adjusted_rewards.squeeze() + self._lambda * self.target_network(next_states).max(dim=1)[0].detach()*(1 - dones)
        policy_q = self.network(states).gather(1, actions)
        loss = F.smooth_l1_loss(policy_q.squeeze(), target_q.squeeze())
        loss.backward()

        self.optimizer.step()

        return loss.to("cpu").detach().numpy(), rnd_loss.to("cpu").detach().numpy()

    # 네트워크 모델 저장
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, save_path + '/ckpt')

    # 학습 기록
    def write_summary(self, score, q_loss, rnd_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/q_loss", q_loss, step)
        self.writer.add_scalar("model/rnd_loss", rnd_loss, step)


# Main 함수 -> 전체적으로 Adversarial PPO 알고리즘을 진행 
if __name__ == '__main__':
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

    agent_A = Agent("A")
    agent_B = Agent("B")

    episode = 0
    q_losses_A, rnd_losses_A, score_A = [], [], 0
    q_losses_B, rnd_losses_B, score_B = [], [], 0
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

            if (step + 1) % train_interval == 0:
                # 학습 수행
                q_loss_A, rnd_loss_A = agent_A.train_model()
                q_losses_A.append(q_loss_A)
                rnd_losses_A.append(rnd_loss_A)

                q_loss_B, rnd_loss_B = agent_B.train_model()
                q_losses_B.append(q_loss_B)
                rnd_losses_B.append(rnd_loss_A)

        if done_A or done_B:
            episode += 1

            mean_rnd_loss_A = np.mean(rnd_losses_A) if len(rnd_losses_A) > 0 else 0
            mean_q_loss_A = np.mean(q_losses_A) if len(q_losses_A) > 0 else 0
            agent_A.write_summary(score_A, mean_q_loss_A, mean_rnd_loss_A, step)

            mean_rnd_loss_B = np.mean(rnd_losses_B) if len(rnd_losses_B) > 0 else 0
            mean_q_loss_B = np.mean(q_losses_B) if len(q_losses_B) > 0 else 0
            agent_B.write_summary(score_B, mean_q_loss_B, mean_rnd_loss_B, step)

            print(f"{episode} Episode / Step: {step} / " + \
                  f"A Score: {score_A:.5f} / B Score: {score_B:.5f} / " + \
                  f"A Q_Loss: {mean_q_loss_A:.4f} / B Q_Loss: {mean_q_loss_B:.4f} / " + \
                  f"A rnd Loss: {mean_rnd_loss_A:.4f} / B rnd Loss: {mean_rnd_loss_B:.4f}")

            mean_q_loss_A, rnd_losses_A, score_A = [], [], 0
            mean_q_loss_B, rnd_losses_B, score_B = [], [], 0

            # 네트워크 모델 저장
            if train_mode and episode % save_interval == 0:
                agent_A.save_model()
                agent_B.save_model()

    agent_A.save_model()
    agent_B.save_model()

    env.close()
