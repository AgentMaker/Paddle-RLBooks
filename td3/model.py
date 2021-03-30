import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Actor(nn.Layer):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        return self.max_action * F.tanh(self.l3(a))

    def select_action(self, state):
        state = paddle.to_tensor(state.reshape(1, -1)).astype('float32')
        return self.forward(state).numpy()[0]

class Critic(nn.Layer):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(paddle.concat([q, action], 1)))

        return self.l3(q)