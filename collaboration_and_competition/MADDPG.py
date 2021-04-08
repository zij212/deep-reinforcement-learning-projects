import numpy as np
import torch
import copy
import torch.nn.functional as F

from DDPG import device, DDPG


class ReplayBuffer:
    def __init__(self, args):
        max_len = args['max_len']
        self.args = args
        self.ptr = 0
        self.size = 0

        # initialize arrays to save s, a, r, s', not_done separately
        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.not_done = []

        for agent_id in range(args['num_agents']):
            state_dim = args['state_dim'][agent_id]
            action_dim = args['action_dim'][agent_id]
            self.state.append(np.zeros((max_len, state_dim)))
            self.action.append(np.zeros((max_len, action_dim)))
            self.reward.append(np.zeros((max_len, 1)))
            self.next_state.append(np.zeros((max_len, state_dim)))
            self.not_done.append(np.zeros((max_len, 1)))

        self.max_len = max_len

    def add(self, state, action, reward, next_state, done):
        for agent_id in range(self.args['num_agents']):
            self.state[agent_id][self.ptr] = state[agent_id]
            self.action[agent_id][self.ptr] = action[agent_id]
            self.reward[agent_id][self.ptr] = reward[agent_id]
            self.next_state[agent_id][self.ptr] = next_state[agent_id]
            self.not_done[agent_id][self.ptr] = 1. - float(done[agent_id])

        # increment pointer and size counter
        self.ptr = (self.ptr + 1) % self.max_len
        self.size = min(self.size + 1, self.max_len)

    def sample_idx(self, batch_size):
        return np.random.randint(low=0, high=self.size, size=batch_size, dtype=int)

    def get_experiences(self, idx):
        return (
            [torch.FloatTensor(self.state[agent_id][idx]).to(device)
             for agent_id in range(self.args['num_agents'])],
            [torch.FloatTensor(self.action[agent_id][idx]).to(device)
                for agent_id in range(self.args['num_agents'])],
            [torch.FloatTensor(self.reward[agent_id][idx]).to(device)
                for agent_id in range(self.args['num_agents'])],
            [torch.FloatTensor(self.next_state[agent_id][idx]).to(device)
                for agent_id in range(self.args['num_agents'])],
            [torch.FloatTensor(self.not_done[agent_id][idx]).to(device)
                for agent_id in range(self.args['num_agents'])],
        )


class MADDPG:
    def __init__(self, args):
        self.args = args
        self.replay_buffer = ReplayBuffer(self.args)
        self.agents = [DDPG(args, i) for i in range(self.args['num_agents'])]

    def sample_action(self, s, noise=0):
        """

        :param s: all agents' states
        :param noise: parameter for noise factory
        :return: all agents' actions
        """
        with torch.no_grad():
            a = []
            for agent_id in range(self.args['num_agents']):
                state = s[agent_id]
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                self.agents[agent_id].actor.eval()
                action = self.agents[agent_id].actor(state)
                self.agents[agent_id].actor.train()
                if noise != 0:
                    action += noise * self.agents[agent_id].noise_factory.noise()
                    action = action.clamp(-self.args['max_action'], self.args['max_action'])
                action = action.cpu().detach().squeeze(0).numpy()
                a.append(action)
        return a

    def train(self):

        batch_size = self.args['batch_size']
        idx = self.replay_buffer.sample_idx(batch_size)
        s, a, r, next_s, not_d = self.replay_buffer.get_experiences(idx)

        for agent in self.agents:

            with torch.no_grad():
                next_a = []
                for i in range(self.args['num_agents']):
                    next_a.append(self.agents[i].actor_target(next_s[i]))
                q_target_next = agent.critic_target(next_s, next_a)
                q_target = r[agent.agent_id] + not_d[agent.agent_id] * self.args['gamma'] * q_target_next

            q_pred = agent.critic(s, a)

            critic_loss = F.mse_loss(q_pred, q_target)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            # TODO: add gradient clipping
            agent.critic_optimizer.step()

            a_ = []
            for i in range(self.args['num_agents']):
                a_.append(self.agents[i].actor(s[i]))

            actor_loss = - agent.critic(s, a_).mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            # TODO: add gradient clipping
            agent.actor_optimizer.step()

            agent.soft_update_actor_critic_target()

    def add_experience(self, s, a, r, ns, d):
        self.replay_buffer.add(s, a, r, ns, d)

    def save(self, filename='models/solution'):
        for agent in self.agents:
            agent_name = f'_agent{agent.agent_id}'
            torch.save(agent.critic.state_dict(), filename + agent_name + '_critic')
            torch.save(agent.critic_optimizer.state_dict(), filename + agent_name + '_critic_optimizer')
            torch.save(agent.actor.state_dict(), filename + agent_name + '_actor')
            torch.save(agent.actor_optimizer.state_dict(), filename + agent_name + '_actor_optimizer')

    def load(self, filename='models/solution'):
        for agent in self.agents:
            agent_name = f'_agent{agent.agent_id}'
            agent.critic.load_state_dict(torch.load(filename + agent_name + '_critic'))
            agent.critic_optimizer.load_state_dict(
                torch.load(filename + agent_name + '_critic_optimizer'))

            agent.actor.load_state_dict(torch.load(filename + agent_name + '_actor'))
            agent.actor_optimizer.load_state_dict(torch.load(filename + agent_name + '_actor_optimizer'))

            agent.critic_target = copy.deepcopy(agent.critic)
            agent.actor_target = copy.deepcopy(agent.actor)

