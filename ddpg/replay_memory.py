import collections
import random

import paddle


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            batch_state.append(s)
            batch_action.append(a)
            batch_reward.append(r)
            batch_next_state.append(s_p)
            batch_done.append(done)
        batch_state = paddle.to_tensor(batch_state, dtype='float32')
        batch_action = paddle.to_tensor(batch_action, dtype='float32')
        batch_reward = paddle.to_tensor(batch_reward, dtype='float32')
        batch_next_state = paddle.to_tensor(batch_next_state, dtype='float32')
        batch_done = paddle.to_tensor(batch_done, dtype='float32')

        return batch_state, batch_action, batch_reward, batch_next_state, batch_done

    def __len__(self):
        return len(self.buffer)
