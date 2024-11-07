import numpy as np
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

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.idx = 0
        self.buffer: list[Experience] = []

    def size(self):
        return len(self.buffer)

    def push(self, experience: Experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.idx] = experience
        self.idx += 1
        if self.idx == self.capacity:
            self.idx = 0
    
    def get_minibatch(self, batch_size: int):
        s = len(self.buffer)
        indices = np.random.choice(np.arange(s), replace=False, size=batch_size)
        selected_experiences = [self.buffer[idx] for idx in indices]

        states = np.vstack([exp.state for exp in selected_experiences])
        actions = np.vstack([exp.action for exp in selected_experiences])
        rewards = np.vstack([exp.reward for exp in selected_experiences])
        next_states = np.vstack([exp.next_state for exp in selected_experiences])
        dones = np.array([int(exp.done) for exp in selected_experiences]).reshape(-1, 1)

        return (states, actions, rewards, next_states, dones)

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, middle_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, middle_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(middle_dim, middle_dim)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(middle_dim, output_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x

class Trainer:
    def __init__(self, model: nn.Module, capacity: int, batch_size: int, gamma: float):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = 0.0001)
        self.replay_buffer = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.gamma = gamma

    def replay(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        (states, actions, rewards, next_states, dones) = self.replay_buffer.get_minibatch(self.batch_size)
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones)

        self.model.eval()
        #: TQ = r + γmaxQ(s', a')
        expected_q_values = self.model(states).gather(1, actions)

        next_states_max_qvalues = self.model(next_states).max(1)[0].detach().unsqueeze(1)

        target_q_values = rewards + self.gamma * (1 - dones) * next_states_max_qvalues
        
        self.model.train()
        loss = F.smooth_l1_loss(expected_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def memorize(self, exp: Experience):
        self.replay_buffer.push(exp)

    def decide_action(self, state: np.ndarray, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                return self.model(torch.from_numpy(state)).max(0)[1].item()
        else:
            return random.randrange(2)

GAMMA = 0.99
ENV = 'CartPole-v1'
CAPACITY = 10000
BATCH_SIZE = 32
MAX_STEPS = 200

class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        self.trainer = Trainer(QNetwork(4, 64, 2), CAPACITY, BATCH_SIZE, GAMMA)
    
    def run(self):
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
                self.trainer.replay()

                state = next_state
                if terminated:
                    print(episode, step)
                    break

env = Environment()
env.run()
