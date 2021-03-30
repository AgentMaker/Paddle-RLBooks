import gym
import numpy as np
from itertools import count

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle.distribution import Categorical

gamma = 0.99
render = True
log_interval = 10

env = gym.make('CartPole-v1')
env.seed(1)
paddle.seed(1)

class Policy(nn.Layer):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, inputs):
        x = F.relu(F.dropout(self.fc1(inputs), 0.6))
        x = self.fc2(x)

        return F.softmax(x, -1)

    def select_action(self, inputs):
        x = paddle.to_tensor(inputs).astype('float32').unsqueeze(0)
        probs = self.forward(x)
        m = Categorical(probs)
        action = m.sample([1])
        self.saved_log_probs.append(m.log_prob(action))

        return action

policy = Policy()
optimizer = optim.Adam(parameters=policy.parameters(), learning_rate=1e-2)
eps = np.finfo(np.float32).eps.item()

def finish_episode():
    R = 0
    policy_loss = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
    returns = paddle.full([len(policy.rewards)], R)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.clear_grad()
    policy_loss = paddle.concat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def main():
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action.numpy()[0][0])
            if render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
            # save model
            paddle.save(policy.state_dict(), 'model/model.pdparams')
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs is {}!".format(running_reward, i_episode))
            break

if __name__ == '__main__':
    main()