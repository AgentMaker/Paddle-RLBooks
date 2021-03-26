import paddle.nn as nn
import paddle.nn.functional as F

class Model(nn.Layer):
    def __init__(self, num_inputs, num_actions, atoms=51):
        super(Model, self).__init__()
        self.num_actions = num_actions
        self.atoms = atoms

        self.conv1 = nn.Conv2D(num_inputs, 32, 3, stride=3)
        self.conv2 = nn.Conv2D(32, 32, 3, stride=3)
        self.conv3 = nn.Conv2D(32, 64, 3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 3 * 2, 256)
        self.fc2 = nn.Linear(256, num_actions * atoms)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return F.softmax(x.reshape((-1, self.num_actions, self.atoms)), axis=2)
