import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        ##### TODO ######
        ### Complete definition
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.fc3 = nn.Linear(128, 1)


    def forward(self, x):
        ##### TODO ######
        ### Complete definition
        xx = F.relu(self.fc1(x))
        action_score = self.fc2(xx)
        state_values = self.fc3(xx)
        return F.softmax(action_score), state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()
reward_log = []
running_reward_log = []
x_axis_reward = []
x_axis_running_reward = []


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    return action

def sample_episode():

    state, ep_reward = env.reset(), 0
    episode = []

    for t in range(1, 10000):  # Run for a max of 10k steps

        action = select_action(state)

        # Perform action
        next_state, reward, done, _ = env.step(action.item())

        episode.append((state, action, reward))
        state = next_state

        ep_reward += reward

        if args.render:
            env.render()

        if done:
            break

    return episode, ep_reward

def compute_losses(episode):

    ####### TODO #######
    #### Compute the actor and critic losses
    #actor_loss, critic_loss = None, None

    r = 0
    actor_loss = 0
    critic_loss = 0
    for i in reversed(range(len(episode))):
        curState = episode[i][0]
        curAction = episode[i][1]
        curReward = episode[i][2]

        r = r*args.gamma + curReward
        probs, value = model(torch.tensor(curState).type(torch.float32))
        advantage = r - value

        actor_loss += -torch.log(probs[curAction]) * advantage
        critic_loss += pow(advantage, 2)

    return actor_loss, critic_loss


def main():
    running_reward = 10
    for i_episode in count(1):

        episode, episode_reward = sample_episode()

        optimizer.zero_grad()

        actor_loss, critic_loss = compute_losses(episode)

        loss = actor_loss + critic_loss

        loss.backward()

        optimizer.step()

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        reward_log.append(episode_reward)
        x_axis_reward.append(i_episode)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, episode_reward, running_reward))
            running_reward_log.append(running_reward)
            x_axis_running_reward.append(i_episode)
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, len(episode)))
            break

    plt.plot(x_axis_reward, reward_log)
    plt.plot(x_axis_running_reward, running_reward_log)
    plt.legend(['Episode reward', 'Running reward'], loc='upper left')
    plt.title('Trainning')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()



if __name__ == '__main__':
    main()
