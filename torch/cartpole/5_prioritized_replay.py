import numpy as np
import math
from itertools import count
from typing import NamedTuple

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

random.seed(768)
torch.manual_seed(768)

device = torch.device('mps')

from dataclasses import dataclass

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, total_steps: int, alpha = 0.6, beta = 0.4):
        self.capacity = capacity
        self.idx = 0
        self.buffer: list[Experience] = []
        self.priority = np.empty(capacity, dtype=np.float32)
        self.max_priority = 1.0
        self.alpha = alpha
        self.beta_scheduler = (lambda steps: beta + (1 - beta) * steps / total_steps)
        self.epsilon = 1e-3

    def size(self):
        return len(self.buffer)

    def push(self, experience: Experience):
        # reward clip
        experience.reward = np.clip(experience.reward, -1, 1)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.idx] = experience
        self.priority[self.idx] = self.max_priority
        self.idx += 1
        if self.idx == self.capacity:
            self.idx = 0
    
    def get_minibatch(self, batch_size: int, steps: int):
        s = len(self.buffer)

        probs = self.priority[0:s] / self.priority[0:s].sum()
        indices = np.random.choice(np.arange(s), p=probs, replace=False, size=batch_size)
        
        beta = self.beta_scheduler(steps)
        weights = (probs[indices] * s) ** (-1 * beta)
        weights /= weights.max()
        weights = weights.reshape(-1, 1).astype(np.float32)

        selected_experiences = [self.buffer[idx] for idx in indices]

        states = np.vstack([exp.state for exp in selected_experiences])
        actions = np.vstack([exp.action for exp in selected_experiences])
        rewards = np.vstack([exp.reward for exp in selected_experiences])
        next_states = np.vstack([exp.next_state for exp in selected_experiences])
        dones = np.array([int(exp.done) for exp in selected_experiences]).reshape(-1, 1)

        return indices, weights, (states, actions, rewards, next_states, dones)

    def update_priority(self, indices, td_errors):
        td_errors = np.abs(td_errors)
        self.priority[indices] = (td_errors + self.epsilon) ** self.alpha
        self.max_priority = max(self.max_priority, td_errors.max())

class FactorizedNoisy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.w_mu = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.w_sigma = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.b_mu = nn.Parameter(torch.Tensor(output_dim))
        self.b_sigma = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w_mu.size(1))
        self.w_mu.data.uniform_(-stdv, stdv)
        self.b_mu.data.uniform_(-stdv, stdv)

        initial_sigma = 0.5 * stdv
        self.w_sigma.data.fill_(initial_sigma)
        self.b_sigma.data.fill_(initial_sigma)

    @staticmethod
    def f_(x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def forward(self, x):
        rand_in = self.f_(torch.randn(1, self.input_dim, device=self.w_mu.device))
        rand_out = self.f_(torch.randn(self.output_dim, 1, device=self.w_mu.device))
        w_epsilon = torch.matmul(rand_out, rand_in)
        b_epsilon = rand_out.squeeze()

        w = self.w_mu + self.w_sigma * w_epsilon
        b = self.b_mu + self.b_sigma * b_epsilon
        return F.linear(x, w, b)

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, middle_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, middle_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(middle_dim, middle_dim)
        self.relu2 = nn.ReLU()
        self.adv = FactorizedNoisy(middle_dim, output_dim)
        self.v = nn.Linear(middle_dim, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        adv = self.adv(x)
        v = self.v(x).expand(-1, adv.size(1))
        output = v + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        return output

class DDQNTrainer:
    def __init__(self, capacity: int, batch_size: int, gamma: float):
        self.main_qnet = QNetwork(4, 64, 2)
        self.target_qnet = QNetwork(4, 64, 2)
        self.optimizer = optim.Adam(self.main_qnet.parameters(), lr = 0.0001)
        self.replay_buffer = PrioritizedReplayBuffer(capacity, 10000)
        self.batch_size = batch_size
        self.gamma = gamma

    def replay(self, steps: int):
        if self.replay_buffer.size() < self.batch_size:
            return
        indices, weights, (states, actions, rewards, next_states, dones) = self.replay_buffer.get_minibatch(self.batch_size, steps)
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones)

        self.main_qnet.eval()
        self.target_qnet.eval()
        expected_q_values = self.main_qnet(states).gather(1, actions)

        #: TQ = r + γ Q_target(s', a'), a' = argmax_a'{ Q_main(s', a') }
        next_actions = self.main_qnet(next_states).max(1)[1].detach().unsqueeze(1)
        next_states_max_qvalues = self.target_qnet(next_states).gather(1, next_actions)
        target_q_values = rewards + self.gamma * (1 - dones) * next_states_max_qvalues
        
        self.main_qnet.train()
        losses = F.smooth_l1_loss(expected_q_values, target_q_values, reduction="none")
        loss = (losses * torch.from_numpy(weights)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.replay_buffer.update_priority(indices, losses.detach().numpy().flatten())

    def sync_target_qnet(self):
        self.target_qnet.load_state_dict(self.main_qnet.state_dict())

    def memorize(self, exp: Experience):
        self.replay_buffer.push(exp)

 
    def decide_action(self, state: np.ndarray, episode):
        self.main_qnet.eval()
        with torch.no_grad():
            return self.main_qnet(torch.from_numpy(state).unsqueeze(0)).max(1)[1].item()
        #epsilon = 0.5 * (1 / (episode + 1))
        # if epsilon <= np.random.uniform(0, 1):
        #     self.main_qnet.eval()
        #     with torch.no_grad():
        #         return self.main_qnet(torch.from_numpy(state).unsqueeze(0)).max(1)[1].item()
        # else:
        #     return random.randrange(2)

GAMMA = 0.99
ENV = 'CartPole-v1'
CAPACITY = 10000
BATCH_SIZE = 32
MAX_STEPS = 200

class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        self.trainer = DDQNTrainer(CAPACITY, BATCH_SIZE, GAMMA)
    
    def run(self):
        total_steps = 0
        for episode in count(0):
            state = self.env.reset()[0]
            for step in range(MAX_STEPS):
                action = self.trainer.decide_action(state, episode)
                next_state, _, terminated, _, _ = self.env.step(action)
                reward = 0.0
                if terminated:
                    reward = -1.0
                elif step >= 195:
                    reward = 1.0
                    terminated = True
                else:
                    reward = 0.0

                self.trainer.memorize(Experience(state, action, reward, next_state, terminated))
                total_steps += 1
                if total_steps % 4 == 0:
                    self.trainer.replay(episode)
                if total_steps % 10 == 0:
                    self.trainer.sync_target_qnet()

                state = next_state
                if terminated:
                    print(episode, step)
                    break

env = Environment()
env.run()
