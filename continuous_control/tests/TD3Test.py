import unittest
from TD3 import ReplayBuffer, Actor, Critic, TD3
import numpy as np
import torch
import random
import os


class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.replay_buffer = ReplayBuffer(state_dim=2, action_dim=3, max_len=100)
    
    def test_add_experience(self):
        state_list = [1., 1.]
        action_list = [1., 1., 1.]
        reward_float = 0.
        next_state_list = [2., 2.]
        done_float = 0.
        
        self.replay_buffer.add(
            state=state_list, action=action_list, reward=reward_float, 
            next_state=next_state_list, done=done_float)
        
        idx = self.replay_buffer.ptr - 1
        
        state = self.replay_buffer.state[idx]
        np.testing.assert_array_equal(state, np.array(state_list))
        
        action = self.replay_buffer.action[idx]
        np.testing.assert_array_equal(action, np.array(action_list))
        
        reward = self.replay_buffer.reward[idx]
        self.assertEqual(reward, reward_float)
        
        next_state = self.replay_buffer.next_state[idx]
        np.testing.assert_array_equal(next_state, np.array(next_state_list))
        
        not_done = self.replay_buffer.not_done[idx]
        self.assertEqual(not_done, 1.0 - done_float)

        
class TestActor(unittest.TestCase):
    
    def setUp(self):
        self.state_dim = 5
        self.action_dim = 2
        self.max_action = 1
        self.batch_size = 50

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action)
    
    def test_actor_output_dimension(self):
        # make sure the actor network's output tensor's dimension is (batch_size, action_dim)
        action = self.actor.forward(
            torch.rand((self.batch_size, self.state_dim)))
        
        self.assertEqual(
            (action.size(0), action.size(1)), 
            (self.batch_size, self.action_dim))
    
    def test_action_range(self):
        # the range for action must be (-self.max_action, self.max_action)
        action = self.actor.forward(
            torch.rand((self.batch_size, self.state_dim)))

        self.assertEqual((action > -self.max_action).all(), True) 
        self.assertEqual((action < self.max_action).all(), True) 


class TestCritic(unittest.TestCase):
    
    def setUp(self):
        self.state_dim, self.action_dim, self.batch_size = 5, 2, 10
        self.critic = Critic(self.state_dim, self.action_dim)
    
    def test_twin_network_forward(self):
        # Critic's forward function must return 2 Q values
        states = torch.rand((self.batch_size, self.state_dim))
        actions = torch.rand((self.batch_size, self.action_dim))
        output = self.critic.forward(states, actions)
        self.assertEqual(len(output), 2)
    
    def test_Q1_forward(self):
        states = torch.rand((self.batch_size, self.state_dim))
        actions = torch.rand((self.batch_size, self.action_dim))
        output = self.critic(states, actions)
        q1_output = self.critic.Q1(states, actions)
        assert torch.equal(output[0], q1_output)
        

class TestTD3Agent(unittest.TestCase):
    
    def setUp(self):
        self.state_dim, self.action_dim, self.max_action = 20, 4, 1
    
    def test_save_and_load_agent_weights(self):
        agent = TD3(self.state_dim, self.action_dim, self.max_action)
        filename = 'test_weights/test'
        agent.save(filename)
        os.path.exists(os.path.join(filename, '_critic'))
        os.path.exists(os.path.join(filename, '_critic_optimizer'))
        os.path.exists(os.path.join(filename, '_actor'))
        os.path.exists(os.path.join(filename, '_actor_optimizer'))
        agent.load(filename)
        
    
    def test_sample_action(self):
        agent = TD3(self.state_dim, self.action_dim, self.max_action)
        action = agent.sample_action(np.random.randn(self.state_dim))
        self.assertEqual(action.size, self.action_dim)
        
    def test_train_agent(self):
        agent = TD3(self.state_dim, self.action_dim, self.max_action)
        replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_len=10)
        batch_size = 5
        for _ in range(batch_size):
            replay_buffer.add(
                state=np.random.rand(self.state_dim), 
                action=np.random.rand(self.action_dim), 
                reward=random.random(), 
                next_state=np.random.rand(self.state_dim), 
                done=0)
        agent.train(replay_buffer=replay_buffer, batch_size=batch_size)
        
        
if __name__ == '__main__':
    unittest.main()