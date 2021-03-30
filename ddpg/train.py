from model import Actor, Critic
from replay_memory import ReplayMemory
import paddle
import paddle.nn.functional as F
import numpy as np
import gym

batch_size = 256
num_episodes = 100000
memory_size = 1000000
learning_rate = 1e-3
gamma = 0.99
ratio = 0.005
update_num = 0

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
actor_optimizer = paddle.optimizer.Adam(parameters=actor.parameters(),
                                  learning_rate=learning_rate)

critic = Critic(state_dim, action_dim)
target_critic = Critic(state_dim, action_dim)
target_critic.eval()
target_critic.load_dict(critic.state_dict())
critic_optimizer = paddle.optimizer.Adam(parameters=critic.parameters(),
                                  learning_rate=learning_rate)

rpm = ReplayMemory(memory_size)

def train():
    global update_num
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
            best_v = target_critic(batch_next_state, target_actor(batch_next_state))
            best_v = batch_reward + (gamma * best_v * (1 - batch_done)).detach()

            current_v = critic(batch_state, batch_action)

            critic_loss = F.mse_loss(current_v, best_v)
            critic_optimizer.clear_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_loss = -critic(batch_state, actor(batch_state)).mean()
            actor_optimizer.clear_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # 指定的训练次数更新一次目标模型的参数
            if update_num % 200 == 0:
                target_actor.load_dict(actor.state_dict())
                target_critic.load_dict(critic.state_dict())
            update_num += 1

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