import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random


BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
LR = 5e-4
GAMMA = 0.99
TAU = 1e-3
UPDATE_EVERY = 4

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, xb):
        return self.network(xb)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience_factory = namedtuple(
            'experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        experience = self.experience_factory(
            state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self):
        if len(self.buffer) < self.batch_size:
            raise Exception('mini batch size is larger than buffer size')

        experiences = random.sample(self.buffer, self.batch_size)
        # indices = np.random.choice(len(self.buffer), self.batch_size)
        # experiences = [self.buffer[i] for i in indices]

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return str(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.name = 'DQNAgent'
        self.state_size = state_size
        self.action_size = action_size
        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        self.q_pred_network = QNetwork(state_size, action_size).to(device)
        self.q_target_network = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_pred_network.parameters(), lr=LR)

        self.t_step = 0

    def sample(self, state, eps):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        if np.random.uniform(0, 1) > eps:
            self.q_pred_network.eval()
            with torch.no_grad():
                action_values = self.q_pred_network(state)
            self.q_pred_network.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.buffer) > BATCH_SIZE:
                experiences = self.buffer.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Q_{target}(S_t,A_t) = r + \gamma * max_{a', a' \in A(S_{t+1})}Q(S_{t+1}, a') * (1 - done)
        Q_next_states = self.q_target_network(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_next_states * (1 - dones))

        # Q_{pred}(S_t, A_t)
        Q_preds = self.q_pred_network(states).gather(1, actions)

        # Minimize the MSE of action-values
        loss = F.mse_loss(Q_preds, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.q_pred_network, self.q_target_network, TAU)

    @staticmethod
    def soft_update(pred_network, target_network, tau):
        for pred_param, target_param in zip(
                pred_network.parameters(), target_network.parameters()):
            target_param.data.copy_(
                (1.0 - tau) * target_param.data + tau * pred_param.data)

