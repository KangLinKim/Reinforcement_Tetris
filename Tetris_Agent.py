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
state_size = [1*3, 224, 224]
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
        self.params = None
        self.compression = 0.5
        self.k = 32
        self.dim = [0, 0, 0]
        # Convolution
        # codes are in the Function - ConvolutionLayer

        # Dense Block(1)
        # codes are in the Function - DenseBlock
        self.denseBlock_1ChannelSize = self.k
        self.denseBlock_1Cnt = 3

        # Transition Layer(1)
        # codes are in the Function - TransitionLayer

        # Dense Block(2)
        # codes are in the Function - DenseBlock
        self.denseBlock_2ChannelSize = self.k
        self.denseBlock_2Cnt = 6

        # Transition Layer(2)
        # codes are in the Function - TransitionLayer

        # Dense Block(3)
        # codes are in the Function - DenseBlock
        self.denseBlock_3ChannelSize = self.k
        self.denseBlock_3Cnt = 24

        # Dense Block(4)
        # codes are in the Function - DenseBlock
        self.denseBlock_4ChannelSize = self.k
        self.denseBlock_4Cnt = 16

        self.flat = torch.nn.Flatten()

        # FC Layers
        self.fc1 = torch.nn.Linear(14464, 1024, device=device)
        self.fc2 = torch.nn.Linear(1024, 256, device=device)
        self.fc_out = torch.nn.Linear(256, action_size, device=device)

    def ConvolutionLayer(self, x):
        conv_kernel_size = 7
        conv_stride = 2
        conv_out_channel = 64
        # padding = ((output_size - 1) * stride + kernel_size - input_size) / 2
        conv_padding = ((state_size[1] // 2 - 1) * conv_stride + conv_kernel_size - state_size[1]) / 2
        if conv_padding > int(conv_padding):
            conv_padding += 0.5
        conv = torch.nn.Conv2d(in_channels=state_size[0], out_channels=conv_out_channel,
                               kernel_size=conv_kernel_size, stride=conv_stride,
                               padding=int(conv_padding), device=device)
        self.dim = (conv_out_channel,
                    (state_size[1] - conv_kernel_size + 2 * conv_padding) // conv_stride + 1,
                    (state_size[2] - conv_kernel_size + 2 * conv_padding) // conv_stride + 1)
        batchNorm = torch.nn.BatchNorm2d(num_features=conv_out_channel, device=device)

        # Pooling
        maxPool_kernel_size = 3
        maxPool_stride = 2
        maxPool_padding = ((self.dim[1] // 2 - 1) * maxPool_stride + maxPool_kernel_size - self.dim[2]) / 2
        if maxPool_padding > int(maxPool_padding):
            maxPool_padding += 0.5
        maxPool = torch.nn.MaxPool2d(kernel_size=maxPool_kernel_size, stride=maxPool_stride,
                                     padding=int(maxPool_padding))

        self.dim = (conv_out_channel, self.dim[1]//2, self.dim[2]//2)

        x = conv(x)
        x = batchNorm(x)
        x = F.leaky_relu(x)
        x = maxPool(x)
        return x

    def DenseBlock(self, x, denseChannelSize):
        conv_1_kernel_size = 1
        conv_1_stride = 1
        conv_1_out_channel = 128
        conv_1_padding = 0
        conv_1 = torch.nn.Conv2d(in_channels=self.dim[0], out_channels=conv_1_out_channel,
                                 kernel_size=conv_1_kernel_size, stride=conv_1_stride,
                                 padding=conv_1_padding, device=device)
        batchNorm_1 = torch.nn.BatchNorm2d(num_features=conv_1_out_channel, device=device)

        conv_2_kernel_size = 3
        conv_2_stride = 1
        conv_2_out_channel = denseChannelSize
        conv_2_padding = 1
        conv_2 = torch.nn.Conv2d(in_channels=conv_1_out_channel, out_channels=conv_2_out_channel,
                                 kernel_size=conv_2_kernel_size, stride=conv_2_stride,
                                 padding=conv_2_padding, device=device)
        batchNorm_2 = torch.nn.BatchNorm2d(num_features=conv_2_out_channel, device=device)

        tmpX = conv_1(x)
        tmpX = batchNorm_1(tmpX)
        tmpX = F.leaky_relu(tmpX)
        tmpX = conv_2(tmpX)
        tmpX = batchNorm_2(tmpX)
        tmpX = F.leaky_relu(tmpX)

        return tmpX

    def TransitionBlock(self, x, current_shape):
        conv_kernel_size = 1
        conv_stride = 1
        conv_out_channel = int(current_shape * self.compression)
        conv_padding = ((self.dim[1] - 1) * conv_stride + conv_kernel_size - self.dim[2]) / 2
        if conv_padding > int(conv_padding):
            conv_padding += 0.5
        convN = torch.nn.Conv2d(in_channels=current_shape, out_channels=conv_out_channel,
                                kernel_size=conv_kernel_size, stride=conv_stride,
                                padding=int(conv_padding), device=device)
        batchNorm = torch.nn.BatchNorm2d(num_features=conv_out_channel, device=device)

        avgPool_kernel_size = 2
        avgPool_stride = 2
        avgPool = torch.nn.AvgPool2d(kernel_size=avgPool_kernel_size, stride=avgPool_stride)

        self.dim = (conv_out_channel, self.dim[1]//2, self.dim[2]//2)

        x = convN(x)
        x = batchNorm(x)
        x = F.leaky_relu(x)
        x = avgPool(x)
        return x

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        x = x.permute(0, 1, 2, 3)
        x = self.ConvolutionLayer(x)    #16

        xCopy = x[:]
        for i in range(self.denseBlock_1Cnt):
            tmpX = self.DenseBlock(xCopy, self.denseBlock_1ChannelSize)
            x = torch.cat((x, tmpX), dim=1)
        # print(x.shape)
        x = self.TransitionBlock(x, self.dim[0] + self.denseBlock_1ChannelSize * self.denseBlock_1Cnt)  #8

        xCopy = x[:]
        for i in range(self.denseBlock_2Cnt):
            tmpX = self.DenseBlock(xCopy, self.denseBlock_2ChannelSize)
            x = torch.cat((x, tmpX), dim=1)
        x = self.TransitionBlock(x, self.dim[0] + self.denseBlock_2ChannelSize * self.denseBlock_2Cnt)  #4

        xCopy = x[:]
        for i in range(self.denseBlock_3Cnt):
            tmpX = self.DenseBlock(xCopy, self.denseBlock_3ChannelSize)
            x = torch.cat((x, tmpX), dim=1)
        # x = self.TransitionBlock(x, self.dim[0] + self.denseBlock_3ChannelSize * self.denseBlock_3Cnt)
        #
        # xCopy = x[:]
        # for i in range(self.denseBlock_4Cnt):
        #     tmpX = self.DenseBlock(xCopy, self.denseBlock_4ChannelSize)
        #     x = torch.cat((x, tmpX), dim=1)

        self.dim = x.shape[1:]

        x = self.flat(x)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc_out(x), dim=-1)

        return x


# PPOAgent 클래스 -> PPO 알고리즘을 위한 다양한 함수 정의
class Agent:
    def __init__(self, id):
        self.learning_rate = 3*1e-7
        self.memory_size = 512
        self.batch_size = 64
        self._lambda = 0.95

        self.save_path = f"{save_path}/{id}"
        self.load_path = f"{load_path}/{id}"

        self.network = DQNNetwork()
        self.target_network = copy.deepcopy(self.network)
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

    # 정책을 통해 행동 결정
    def get_action(self, state, training=True):
        # 네트워크 모드 설정
        self.network.train(training)

        # 네트워크 연산에 따라 행동 결정
        q = self.network(torch.FloatTensor(state).to(device))
        # print(list(self.network.parameters()))
        if np.random.rand() <= self.epsilon:
            if np.random.rand() >= 0.5:
                action = torch.multinomial(q, num_samples=1).cpu().numpy()
            else:
                action = np.random.randint(0, action_size, size=(1, 1))

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
        states = torch.as_tensor(states, dtype=torch.float, device=device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=device).view(samples_size, -1)
        rewards = torch.as_tensor(rewards, dtype=torch.float, device=device).view(samples_size, -1)
        next_states = torch.as_tensor(next_states, dtype=torch.float, device=device)
        dones = torch.as_tensor(dones, dtype=torch.float, device=device)

        target_q = rewards.squeeze() + self._lambda * self.target_network(next_states).max(dim=1)[0].detach()*(1 - dones)
        policy_q = self.network(states).gather(1, actions)
        loss = torch.sum(torch.square(policy_q.squeeze() - target_q.squeeze()), dim=1).mean()
        loss.backward()

        self.optimizer.step()

        return loss.to("cpu").detach().numpy()

    # 네트워크 모델 저장
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, save_path + '/ckpt')

    # 학습 기록
    def write_summary(self, score, q_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/q_loss", q_loss, step)


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
    q_losses_A, score_A = [], 0
    q_losses_B, score_B = [], 0
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
                q_loss_A = agent_A.train_model()
                q_losses_A.append(q_loss_A)

                q_loss_B = agent_B.train_model()
                q_losses_B.append(q_loss_B)

        if done_A or done_B:
            episode += 1
            A_q_mean = np.mean(q_losses_A)
            B_q_mean = np.mean(q_losses_B)

            print(f"{episode} Episode / Step: {step} / " + \
                  f"A Score: {score_A:.5f} / B Score: {score_B:.5f} / " + \
                  f"A Q_Loss: {A_q_mean:.4f} / B Q_Loss: {B_q_mean:.4f}")

            agent_A.write_summary(score_A, A_q_mean, step)
            agent_B.write_summary(score_B, B_q_mean, step)

            score_A = 0
            score_B = 0

            # 네트워크 모델 저장
            if train_mode and episode % save_interval == 0:
                agent_A.save_model()
                agent_B.save_model()

    agent_A.save_model()
    agent_B.save_model()

    env.close()
