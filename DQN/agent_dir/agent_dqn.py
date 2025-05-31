import random
import numpy as np
import math
import torch
from torch import nn, optim
from agent_dir.agent import Agent
from collections import deque
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)
        # 权重正交初始化
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class ReplayBuffer:
    def __init__(self, buffer_size, alpha=0.6):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha  # 决定优先级的程度
        
    def __len__(self):
        return len(self.buffer)
    
    def push(self, *transition):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(transition)
        self.priorities.append(max_priority)
    
    def sample(self, batch_size, beta=0.4):
        # 计算采样概率
        if len(self.buffer) == 0:
            return [], [], []
        
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引和计算重要性权重
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # 归一化权重
        
        return samples, indices, weights
    def update_priorities(self, indices, priorities):
        # 确保priorities是标量值
        for idx, priority in zip(indices, priorities.flatten() if hasattr(priorities, 'flatten') else priorities):
            if idx < len(self.priorities):
                # 将优先级作为标量值存储
                self.priorities[idx] = float(priority)
        
    def clean(self):
        self.buffer.clear()
        self.priorities.clear()

class AgentDQN(Agent):
    def __init__(self, env, args):
        super(AgentDQN, self).__init__(env)
        self.args = args
        self.device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(args.input_size, args.hidden_size, env.action_space.n).to(self.device)
        self.target_q_network = QNetwork(args.input_size, args.hidden_size, env.action_space.n).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict()) # 复制权重
        self.target_q_network.eval()  # 将目标网络设置为评估模式

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.lr)
        self.replay_buffer = ReplayBuffer(args.buffer_size)

        self.total_steps = 0 # 总步数计数器
        self.epsilon = args.epsilon_start # 初始化 epsilon
        
        self.convergence_threshold = self.args.convergence_threshold # 收敛阈值
        self.consecutive_threshold = self.args.consecutive_episodes # 连续收敛阈值

    def init_game_setting(self):
        pass

    def train(self):
        if len(self.replay_buffer) < self.args.batch_size:
            return
        
        # 使用优先经验回放
        transitions, indices, weights = self.replay_buffer.sample(self.args.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(np.array(batch[1])).unsqueeze(1).to(self.device) #TODO
        reward_batch = torch.FloatTensor(np.array(batch[2])).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch[4])).unsqueeze(1).to(self.device)

        q_values = self.q_network(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_actions = self.q_network(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_q_network(next_state_batch).gather(1, next_state_actions)

        expected_q_values = reward_batch + (self.args.gamma * next_q_values * (1 - done_batch))
        
        # 计算TD误差
        td_errors = torch.abs(q_values - expected_q_values).detach().cpu().numpy()
        # 更新优先级
        new_priorities = td_errors + 1e-6  # 添加小常数防止优先级为0
        self.replay_buffer.update_priorities(indices, new_priorities)

        # 加权Huber损失
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        loss = (weights * nn.functional.smooth_l1_loss(q_values, expected_q_values, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.args.grad_norm_clip) # 梯度裁剪
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.args.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            
        return loss.item()
    
    def make_action(self, observation, test=True): #TODO
        """
        返回智能体的预测动作
        输入: observation (观察值)
        返回: action (动作)
        """
        if not test: # 训练时的 Epsilon-greedy 策略
            if random.random() < self.epsilon: 
                return self.env.action_space.sample()
        
        # 测试时或不进行探索时的贪婪动作
        with torch.no_grad(): # 在此上下文中不计算梯度
            observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device) # 将观察值转换为张量并移至设备
            q_values = self.q_network(observation) 
            return q_values.max(1)[1].item() 
        
    def run(self):
        state = self.env.reset()
        episode_reward = 0
        episode_count = 0

        # 画图记录参数
        rewards = []

        while self.total_steps < self.args.n_frames:
            if not self.args.test:
                self.epsilon = self.args.epsilon_end + (self.args.epsilon_start - self.args.epsilon_end) * \
                 math.exp(-1. * self.total_steps / self.args.epsilon_decay)
            # TODO
            action = self.make_action(state, test=self.args.test)
            next_state, reward, done, _ = self.env.step(action)

            if not self.args.test: # 如果不是测试模式
                self.replay_buffer.push(state, action, reward, next_state, done) # 将转换存储到经验回放缓冲区
                self.train() # 训练模型

            state = next_state # 更新当前状态
            episode_reward += reward #累积回合奖励
            self.total_steps += 1 # 增加总步数

            if done: # 如果回合结束
                episode_count += 1 
                rewards.append(episode_reward)

                print(f"Episode {episode_count},total_steps:{self.total_steps},Total reward: {episode_reward},epsilon: {self.epsilon:.4f}")

                # 重置状态和奖励
                state = self.env.reset()
                episode_reward = 0

        self.plot(rewards)
        return self
    
    def plot(self, acc_rewards):
        plt.figure(figsize=(10, 5))
        acc_rewards = np.array(acc_rewards)
        plt.plot(acc_rewards, label='Reward for each episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Rewards')
        plt.legend()
        plt.grid()
        plt.show()

