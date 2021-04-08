import torch
import torch.nn as nn
import copy
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.network = nn.Sequential(
            nn.BatchNorm1d(state_dim),
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, action_dim),
            nn.Tanh())

    def forward(self, state):
        action = self.network(state)
        return self.max_action * action


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.normalize_state = nn.BatchNorm1d(state_dim)
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, state, action):
        s = torch.cat(state, dim=-1)
        normalized_s = self.normalize_state(s)
        a = torch.cat(action, dim=-1)
        x = torch.cat([normalized_s, a], dim=-1)
        q = self.network(x)
        return q


class OUNoise:

    def __init__(self, action_dim, scale=0.1, mu=0, theta=0.2, sigma=0.2):
        self.action_dim = action_dim
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        self.state = torch.ones(self.action_dim).to(device) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.tensor(np.random.randn(len(x))).float().to(device)
        self.state = x + dx
        return self.state * self.scale


class DDPG:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id

        self.max_action = args['max_action']
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.update_freq = args['update_freq']

        self.noise_factory = OUNoise(
            action_dim=args['action_dim'][agent_id],
            scale=args['noise_scale'],
            mu=0,
            theta=args['noise_theta'],
            sigma=args['noise_sigma'])

        self.actor = Actor(
            args['state_dim'][agent_id],
            args['action_dim'][agent_id],
            self.max_action
        )
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args['actor_lr'])

        self.critic = Critic(
            sum(args['state_dim']),
            sum(args['action_dim'])
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args['critic_lr'])

    def soft_update_actor_critic_target(self):
        tau = self.args['tau']

        with torch.no_grad():
            for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()):
                target_param.copy_((1 - tau) * target_param.detach() + tau * param.detach())

            for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()):
                target_param.copy_((1 - tau) * target_param.detach() + tau * param.detach())

