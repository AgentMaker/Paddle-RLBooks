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

def policy_evaluation(env, policy, gamma = 1.0):
    v = np.zeros(env.observation_space.n)
    eps = 1e-10
    while True:
        v_ = np.copy(v)
        for state in range(env.observation_space.n):
            policy_action = policy[state]
            v[state] = sum([prob * (reward + gamma * v_[state_]) for prob, state_, reward, _ in env.env.P[state][policy_action]])
        if (np.sum(np.fabs(v_ - v)) <= eps):
            break

    return v


def policy_improvement(env, v, gamma = 1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        q = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            q[action] = sum([prob * (reward + gamma * v[state_]) for prob, state_, reward, _ in env.env.P[state][action]])

        policy[state] = np.argmax(q)

    return policy


def policy_iteration(env, gamma=1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n))
    max_iterations = 200000
    gamma = 1.0
    for i in range(max_iterations):
        old_v = policy_evaluation(env, policy, gamma)
        new_policy = policy_improvement(env, old_v, gamma)
        if (np.all(policy == new_policy)):
            print('Policy-Iteration converged at step %d.' % (i + 1))
            break
        policy = new_policy

    return policy


if __name__ == '__main__':
    env_name  = 'FrozenLake8x8-v0' # 'FrozenLake-v0'
    env = gym.make(env_name)
    optimal_policy = policy_iteration(env, gamma = 1.0)
    score = evaluate_policy(env, optimal_policy, gamma = 1.0)
    print('Average scores = ', score)
    test_episode(env, optimal_policy)

