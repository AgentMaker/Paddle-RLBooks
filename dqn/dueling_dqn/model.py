import paddle.nn as nn
import paddle.nn.functional as F

class Model(nn.Layer):
    def __init__(self, num_inputs, num_actions):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2D(num_inputs, 32, 3, stride=3)
        self.conv2 = nn.Conv2D(32, 32, 3, stride=3)
        self.conv3 = nn.Conv2D(32, 64, 3, stride=1)
        self.flatten = nn.Flatten()
        self.adv1 = nn.Linear(64 * 3 * 2, 256)
        self.adv2 = nn.Linear(256, num_actions)

        self.val1 = nn.Linear(64 * 3 * 2, 256)
        self.val2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()
