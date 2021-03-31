from model import Actor, Critic
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
exploration_noise = 0.1
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
target_actor = Actor(state_dim, action_dim, max_action)
target_actor.eval()
target_actor.load_dict(actor.state_dict())
actor_optimizer = paddle.optimizer.RMSProp(parameters=actor.parameters(),
                                  learning_rate=learning_rate)

critic_1 = Critic(state_dim, action_dim)
target_critic_1 = Critic(state_dim, action_dim)
target_critic_1.eval()
target_critic_1.load_dict(critic_1.state_dict())
critic_2 = Critic(state_dim, action_dim)
target_critic_2 = Critic(state_dim, action_dim)
target_critic_2.eval()
target_critic_2.load_dict(critic_2.state_dict())
critic_1_optimizer = paddle.optimizer.RMSProp(parameters=critic_1.parameters(),
                                  learning_rate=learning_rate)
critic_2_optimizer = paddle.optimizer.RMSProp(parameters=critic_2.parameters(),
                                  learning_rate=learning_rate)

rpm = ReplayMemory(memory_size)

def train():
    global epoch
    total_reward = 0
    # 重置游戏状态
    state = env.reset()
    while True:
        action = actor.select_action(state)
        noisy = paddle.normal(0, exploration_noise, shape=[env.action_space.shape[0]]).clip(env.action_space.low, env.action_space.high)
        action = (action + noisy).clip(env.action_space.low, env.action_space.high).numpy()

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
            best_v_1 = target_critic_1(batch_next_state, target_actor(batch_next_state))
            best_v_2 = target_critic_2(batch_next_state, target_actor(batch_next_state))
            best_v = paddle.min(paddle.concat([best_v_1, best_v_2], axis=1), axis=1, keepdim=True)
            best_v = batch_reward + (gamma * best_v * (1 - batch_done)).detach()

            current_v_1 = critic_1(batch_state, batch_action)
            critic_loss = F.mse_loss(current_v_1, best_v)
            critic_1_optimizer.clear_grad()
            critic_loss.backward()
            critic_1_optimizer.step()

            current_v_2 = critic_2(batch_state, batch_action)
            critic_loss = F.mse_loss(current_v_2, best_v)
            critic_2_optimizer.clear_grad()
            critic_loss.backward()
            critic_2_optimizer.step()

            if epoch % policy_delay == 0:
                actor_loss = -critic_1(batch_state, actor(batch_state)).mean()
                actor_optimizer.clear_grad()
                actor_loss.backward()
                actor_optimizer.step()

            # 指定的训练次数更新一次目标模型的参数
            if epoch % 200 == 0:
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.set_value(target_param * (1.0 - ratio) + param * ratio)
                for target_param, param in zip(target_critic_1.parameters(), critic_1.parameters()):
                    target_param.set_value(target_param * (1.0 - ratio) + param * ratio)
                for target_param, param in zip(target_critic_2.parameters(), critic_2.parameters()):
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
        paddle.save(critic_1.state_dict(), 'model/critic_1.pdparams')
        paddle.save(critic_2.state_dict(), 'model/critic_2.pdparams')