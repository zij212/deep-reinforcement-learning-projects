import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    
    def __init__(self, state_dim, action_dim, max_len=int(1e6)):
        self.max_len = max_len
        self.ptr = 0
        self.size = 0
        
        # initialize arrays to save s, a, r, s', not_done separately
        self.state = np.zeros((max_len, state_dim))
        self.action = np.zeros((max_len, action_dim))
        self.reward = np.zeros((max_len, 1))
        self.next_state = np.zeros((max_len, state_dim))
        self.not_done = np.zeros((max_len, 1))
    
    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state  
        self.not_done[self.ptr] = 1. - done
        
        # increment pointer and size counter
        self.ptr = (self.ptr + 1) % self.max_len
        self.size = min(self.size + 1, self.max_len)
    
    def sample(self, batch_size):
        
        # sample a batch of experiences tuples and move to the appropriate device
        
        idx = np.random.randint(low=0, high=self.size, size=batch_size, dtype=int)
        
        return (
            torch.FloatTensor(self.state[idx]).to(device),
            torch.FloatTensor(self.action[idx]).to(device),
            torch.FloatTensor(self.reward[idx]).to(device),
            torch.FloatTensor(self.next_state[idx]).to(device),
            torch.FloatTensor(self.not_done[idx]).to(device),
        )

    
class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        
        self.max_action = max_action
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh())
    
    def forward(self, states):
        actions = self.network(states)
        return self.max_action * actions


class Critic(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.network1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1))
        
        self.network2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1))
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], 1)
        q1 = self.network1(x)
        q2 = self.network2(x)
        return q1, q2
    
    def Q1(self, states, actions):
        x = torch.cat([states, actions], 1)
        q1 = self.network1(x)
        return q1


class TD3:
    
    def __init__(self, state_dim, action_dim, max_action, 
                 gamma=0.99, tau=0.005, action_noise=0.2, noise_clip=0.5, update_freq=2):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        # discount factor
        self.gamma = gamma
        # soft update factor
        self.tau = tau
        self.action_noise = action_noise
        self.noise_clip = noise_clip
        self.update_freq = update_freq
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # initialize actor target and critic target
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.iteration = 0
    
    def sample_action(self, state):
        # given a state, samples a deterministic action
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().detach().squeeze(0).numpy()
    
    def train(self, replay_buffer, batch_size=100):
        """
        Steps:
        1. Samples a batch of experience tuples (s, a, r, s', not_done_bool)
        2. Critic learns by minimizing the TD error
            2.1 Calculate the TD target
                With gradient calculation turned off,
                - Use actor target to select next action a'
                  add noise and noise clip to a' for exploration
                - Use critic target to get TD target = r + gamma * not_done_bool * Q(s', a')
                  Here Q(s', a') is the minimum Q(s', a') of the twined critic target networks as 
            2.2 Calculate the prediction
                Get the twined prediction from the critic.
            2.3 Calculate the loss and backpropogate
                Loss = MSE(prediction 1 - TD target) + MSE(prediction 2 - TD Target)
        3. Actor learns by maxmizing the expected return (state-action value) every 
           update_frequency steps  
            3.1 Calculate actor loss and backpropagate
                - Use the actor to select action a^ for current state s
                - Use 1 of the twined critic to calculate Q1(s, a^)
                - Minimize -mean(Q1(s, a^))
            3.2 Soft update actor target with the actor's weight; 
                soft update critic target with the critic's weight
            
        """
        self.iteration += 1
        
        # 1. samples a batch of experience tuple
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)
        
        # 2.1 calculate the target of Q(s, a)
        with torch.no_grad():

            noise = (
                torch.randn_like(action) * self.action_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(state) + noise
            ).clamp(-self.max_action, self.max_action)
            
            q1_target, q2_target = self.critic_target(next_state, next_action)
            td_target = reward + not_done * self.gamma * torch.min(q1_target, q2_target)
        
        # 2.2 calculate the twined predictions of Q(s, a)
        q1_pred, q2_pred = self.critic(state, action)
        
        # 2.3 Minimize TD error and update critic's weight
        critic_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        if self.iteration % self.update_freq == 0:
            
            # 3.1 Calculate actor loss and back propagate
            actor_loss = - self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 3.2 Soft update critic with critic target;
            #     soft update actor with actor target
            with torch.no_grad():
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()):
                    target_param.copy_((1- self.tau) * target_param.detach() + self.tau * param.detach())

                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()):
                    target_param.copy_((1- self.tau) * target_param.detach() + self.tau * param.detach())
    
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + '_critic')
        torch.save(self.critic_optimizer.state_dict(), filename + '_critic_optimizer')
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.actor_optimizer.state_dict(), filename + '_actor_optimizer')
    
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + '_critic'))
        self.critic_optimizer.load_state_dict(torch.load(filename + '_critic_optimizer'))
        
        self.actor.load_state_dict(torch.load(filename + '_actor'))
        self.actor_optimizer.load_state_dict(torch.load(filename + '_actor_optimizer'))
        
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)