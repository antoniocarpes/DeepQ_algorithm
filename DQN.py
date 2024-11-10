#import packages for the project
import torch
from torch import nn
import gym
from collections import deque
import itertools
import numpy as np
import random


#let us first of all define some hyperparameters
gamma=0.99 #discount factor
batch_size=32
buffer_size=50000
min_replay_size=1000
epsilon_start=1.0
epsilon_end=0.01
epsilon_steps=10000
target_update_freq=1000


#now we create the DQN nn class, which inherits from PyTorch neural network class

class DeepQNetwork(nn.Module):
    def __init__(self,env):
        super().__init__()
        in_features=np.prod(env.observation_space.shape)
        self.net=nn.Sequential(nn.Linear(in_features, 64), nn.Tanh(),
                                nn.Linear(64,env.action_space.n))
    
    def forward(self,x):
        return self.net(x)
    
    def act(self,obs):
        obs_t=torch.as_tensor(obs,dtype=torch.float32)
        q_values=self(obs_t.unsqueeze(0))
        max_q_index=torch.argmax(q_values,dim=1)[0]
        action=max_q_index.detach().item()
        return action

#now let us create an environment where we can try our DQN algorithm
#env=gym.make('CartPole-v1',render_mode="rgb_array")
env = gym.make("CartPole-v1")
env.reset()


replay_buffer=deque(maxlen=buffer_size)
rew_buffer=deque([0.01],maxlen=100)#in a deque, when elements are added that overpass the maxlen
#of the deque, the oldest element is eliminated to make room for new appending elements
#this data structure is useful for algorithms that have a fixed memory

episode_reward=0.0

#now that the class for the DQN network is defined we need to define the objects
    
online_net=DeepQNetwork(env)
target_net=DeepQNetwork(env)

#set the parameters of the target_net equal to the parameters of the online_net

target_net.load_state_dict(online_net.state_dict())

optimizer=torch.optim.Adam(online_net.parameters(),lr=5e-4)

#initialize replay buffer
obs,_=env.reset()
#print(obs)
#print('_______________________')
for _ in range(min_replay_size):
    env.render()
    action=env.action_space.sample()
    new_obs, rew, terminated, truncated, _ = env.step(action)
    done=truncated or terminated
    transition=(obs,action,rew,done,new_obs)
    #print('initialization')
    #print(transition)
    #print('__________________________')
    replay_buffer.append(transition)
    #print(new_obs)
    #print('__________________________')
    obs=new_obs
    if done:
        env.reset()

#Now let us go to the main training loop, i.e., where the neural network is trained
obs,_=env.reset()
for step in itertools.count():#basically this loop iterates infinitely
    epsilon=np.interp(step,[0,epsilon_steps],[epsilon_start,epsilon_end])
    random_value=random.random()
    if random_value<=epsilon:
        action=env.action_space.sample()
    else:
        action=online_net.act(obs)
        new_obs,rew,terminated, truncated,_=env.step(action)
        done=truncated or terminated
        transition=(obs,action,rew,done,new_obs)
        #print('training loop')
        #print(transition)
        #print('_____________________________')
        replay_buffer.append(transition)
        obs=new_obs
        episode_reward+=rew
        if done:
            env.reset()
            rew_buffer.append(episode_reward)
            episode_reward=0.0
        #once solved, watch it play
        #if len(rew_buffer)>=100:
         #   if np.mean(rew_buffer)>195:
         #       while True:
          #          action=online_net.act(obs)
           #         obs,_,terminated, truncated,_=env.step(action)
            #        env.render()
             #       if truncated or terminated:
              #          env.reset()
    
    #here we start gradient step
    #pick a batch size number of transitions from the replay buffer in a random way.

    transitions=random.sample(replay_buffer,batch_size)
    observations=np.asarray([t[0] for t in transitions])
    actions=np.asarray([t[1] for t in transitions])
    rews=np.asarray([t[2] for t in transitions])
    dones=np.asarray([t[3] for t in transitions])
    new_observations=np.asarray([t[4] for t in transitions])

    #convert these arrays into pytorch tensors
    observations_t=torch.as_tensor(observations,dtype=torch.float32)
    actions_t=torch.as_tensor(actions,dtype=torch.int64).unsqueeze(-1)
    rews_t=torch.as_tensor(rews,dtype=torch.float32).unsqueeze(-1)
    dones_t=torch.as_tensor(dones,dtype=torch.float32).unsqueeze(-1)
    new_observations_t=torch.as_tensor(new_observations,dtype=torch.float32)

    #compute targets
    target_q_values=target_net(new_observations_t)
    max_target_q_values=target_q_values.max(dim=1, keepdim=True)[0]
    targets=rews_t+gamma*(1-dones_t)*max_target_q_values

    #compute loss
    q_values=online_net(observations_t)
    action_q_values=torch.gather(input=q_values,dim=1,index=actions_t)
    loss=nn.functional.smooth_l1_loss(action_q_values,targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #update target network
    if step % target_update_freq==0:
        target_net.load_state_dict(online_net.state_dict())

    #logging
    if step % 1000 == 0:
        print('Step:', step)
        print('Average reward:',np.mean(rew_buffer))