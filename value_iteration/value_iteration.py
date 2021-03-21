import numpy as np
import gym

def run_episode(env, policy, gamma = 1.0, render = False):
    observation = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        observation, reward, done, _ = env.step(int(policy[observation]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break

    return total_reward

def test_episode(env, policy):
    observation = env.reset()
    while True:
        env.render()
        observation, _, done, _ = env.step(int(policy[observation]))
        if done:
            break

def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]

    return np.mean(scores)

def policy_extraction(env, v, gamma = 1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        q = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for prob, state_, reward, _ in env.env.P[state][action]:
                q[action] += prob * (reward + gamma * v[state_])
        policy[state] = np.argmax(q)

    return policy

def value_iteration(env, gamma = 1.0):
    v = np.zeros(env.observation_space.n)
    max_iterations = 200000
    eps = 1e-5
    for i in range(max_iterations):
        v_ = np.copy(v)
        for state in range(env.observation_space.n):
            q = [sum([prob * (reward + gamma * v_[state_]) for prob, state_, reward, _ in env.env.P[state][action]]) for action in range(env.action_space.n)]
            v[state] = max(q)

        if (np.sum(np.fabs(v_ - v)) <= eps):
            print('Value-iteration converged at iteration %d.' % (i + 1))
            break

    return v

if __name__ == '__main__':
    env_name = 'FrozenLake8x8-v0'  # 'FrozenLake-v0'
    env = gym.make(env_name)
    optimal_v = value_iteration(env, gamma = 1.0)
    policy = policy_extraction(env, optimal_v, gamma = 1.0)
    score = evaluate_policy(env, policy, gamma = 1.0)
    print('Average scores = ', np.mean(score))
    test_episode(env, policy)




