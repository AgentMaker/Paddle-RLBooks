from model import Actor, Critic, Q
from replay_memory import ReplayMemory
import paddle
import paddle.nn.functional as F
import numpy as np
import gym

batch_size = 256
num_episodes = 100000
memory_size = 1000000
policy_delay = 2
learning_rate = 0.1
gamma = 0.99
ratio = 0.005
exploration_noise = 1e-3
epoch = 0

env = gym.make('Pendulum-v0')
env.seed(1)
paddle.seed(1)
np.random.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_val = paddle.to_tensor(1e-7).astype('float32')

actor = Actor(state_dim, action_dim, max_action)
actor_optimizer = paddle.optimizer.RMSProp(parameters=actor.parameters(),
                                  learning_rate=learning_rate)

Q_net = Q(state_dim, action_dim)
Q_optimizer = paddle.optimizer.RMSProp(parameters=Q_net.parameters(),
                                  learning_rate=learning_rate)

critic = Critic(state_dim)
target_critic = Critic(state_dim)
target_critic.eval()
target_critic.load_dict(critic.state_dict())
critic_optimizer = paddle.optimizer.RMSProp(parameters=critic.parameters(),
                                  learning_rate=learning_rate)

rpm = ReplayMemory(memory_size)

def train():
    global epoch
    total_reward = 0
    # 重置游戏状态
    state = env.reset()
    while True:
        action = actor.select_action(state)

        next_state, reward, done, info = env.step(action)
        env.render()
        rpm.append((state, action, reward, next_state, np.float(done)))

        state = next_state
        if done:
            break
        total_reward += reward

        if len(rpm) > batch_size:
            # 获取训练数据
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = rpm.sample(batch_size)
            # 计算损失函数
            expected_Q = Q_net(batch_state, batch_action)
            expected_value = critic(batch_state)
            new_action, log_prob, z, mean, log_std = actor.get_action(batch_state)

            target_value = target_critic(batch_next_state)
            next_q_value = batch_reward + (1 - batch_done) * gamma * target_value
            Q_loss = F.mse_loss(expected_Q, next_q_value.detach())

            expected_new_Q = Q_net(batch_state, new_action)
            next_value = expected_new_Q - log_prob
            value_loss = F.mse_loss(expected_value, next_value.detach())

            log_prob_target = expected_new_Q - expected_value
            policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

            Q_loss.backward()
            Q_optimizer.step()
            Q_optimizer.clear_grad()

            value_loss.backward()
            critic_optimizer.step()
            critic_optimizer.clear_grad()

            policy_loss.backward()
            actor_optimizer.step()
            actor_optimizer.clear_grad()
            # 指定的训练次数更新一次目标模型的参数
            if epoch % 200 == 0:
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.set_value(target_param * (1.0 - ratio) + param * ratio)
            epoch += 1

    return total_reward

if __name__ == '__main__':
    episode = 0
    while episode < num_episodes:
        for t in range(50):
            train_reward = train()
            episode += 1
            print('Episode: {}, Reward: {:.2f}'.format(episode, train_reward))

        paddle.save(actor.state_dict(), 'model/actor.pdparams')
        paddle.save(critic.state_dict(), 'model/critic.pdparams')