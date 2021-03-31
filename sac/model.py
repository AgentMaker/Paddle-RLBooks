import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distribution import Normal

class Actor(nn.Layer):
    def __init__(self, state_dim, action_dim, max_action, log_min_std=-20, log_max_std=2):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.mean = nn.Linear(300, action_dim)
        self.std = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.log_min_std = log_min_std
        self.log_max_std = log_max_std

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = F.relu(self.std(x))
        log_std = paddle.clip(log_std, self.log_min_std, self.log_max_std)

        return mean, log_std

    def select_action(self, state):
        state = paddle.to_tensor(state.reshape(1, -1)).astype('float32')
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.sample([1])
        action = self.max_action * F.tanh(z).detach().numpy()[0].reshape([-1])

        return action

    def get_action(self, state):
        epsilon = paddle.to_tensor(1e-7, dtype='float32')

        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.sample([1])
        action = paddle.tanh(z)
        log_prob = normal.log_prob(z) - paddle.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

class Critic(nn.Layer):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        return self.l3(x)

class Q(nn.Layer):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state = paddle.reshape(state, (-1, self.state_dim))
        action = paddle.reshape(action, (-1, self.action_dim))
        x = paddle.concat([state, action], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
