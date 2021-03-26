import cv2
import numpy as np
import paddle

import flappy_bird.wrapped_flappy_bird as flappyBird
from model import Model

resize_shape = (1, 36, 52)  # 训练缩放的大小
save_model_path = "models/model.pdparams"  # 保存模型路径

# 图像预处理
def preprocess(observation):
    observation = observation[:observation.shape[0] - 100, :]
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    observation = cv2.resize(observation, (resize_shape[1], resize_shape[2]))
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    observation = np.expand_dims(observation, axis=0)
    observation = observation / 255.0
    return observation


def main():
    # 初始化游戏
    env = flappyBird.GameState()
    # 图像输入形状和动作维度
    obs_dim = resize_shape[0]
    action_dim = env.action_dim

    # 创建模型
    model = Model(obs_dim, action_dim)
    model.load_dict(paddle.load(save_model_path))
    model.eval()

    # 开始游戏
    obs = env.reset()
    episode_reward = 0
    done = False
    # 游戏未结束执行一直执行游戏
    while not done:
        obs = preprocess(obs)
        obs = np.expand_dims(obs, axis=0)
        obs = paddle.to_tensor(obs, dtype='float32')
        action = model(obs)
        action = paddle.argmax(action).numpy()[0]
        obs, reward, done, info = env.step(action, is_train=False)
        episode_reward += reward
    print("最终得分为：{:.2f}".format(episode_reward))


if __name__ == '__main__':
    main()
