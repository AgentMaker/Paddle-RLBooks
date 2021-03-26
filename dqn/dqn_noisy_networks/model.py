import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Assign
import math

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features)
        sigma_init = sigma_zero / math.sqrt(in_features)
        sigma_weight = self.create_parameter(
            shape=[in_features, out_features],
            default_initializer=Assign(
                paddle.full((in_features, out_features), sigma_init)
            )
        )
        self.add_parameter("sigma_weight", sigma_weight)
        self.register_buffer("epsilon_input", paddle.zeros((1, in_features)))
        self.register_buffer("epsilon_output", paddle.zeros((out_features, 1)))
        if bias:
            sigma_bias = self.create_parameter(
                shape=[out_features],
                default_initializer=Assign(
                    paddle.full([out_features], sigma_init)
                )
            )
            self.add_parameter("sigma_bias", sigma_bias)

    def _scale_noise(self, shape):
        x = paddle.randn(shape)
        return x.sign().multiply(x.abs().sqrt())

    def forward(self, inputs):
        with paddle.no_grad():
            eps_in = self._scale_noise(self.epsilon_input.shape)
            eps_out = self._scale_noise(self.epsilon_output.shape)
            noise_v = paddle.multiply(eps_in, eps_out).detach()
        return F.linear(inputs, self.weight + self.sigma_weight * noise_v.t(), self.bias + self.sigma_bias * eps_out.squeeze().t())


class Model(nn.Layer):
    def __init__(self, num_inputs, num_actions):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2D(num_inputs, 32, 3, stride=3)
        self.conv2 = nn.Conv2D(32, 32, 3, stride=3)
        self.conv3 = nn.Conv2D(32, 64, 3, stride=1)
        self.flatten = nn.Flatten()
        self.linear = NoisyLinear(64 * 3 * 2, 256)
        self.fc = NoisyLinear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.linear(x)
        return self.fc(x)
