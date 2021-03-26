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
        batch_obs, batch_action, batch_reword, batch_next_obs, batch_done = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, isOver = experience
            batch_obs.append(s)
            batch_action.append(a)
            batch_reword.append(r)
            batch_next_obs.append(s_p)
            batch_done.append(isOver)
        batch_obs = paddle.to_tensor(batch_obs, dtype='float32')
        batch_action = paddle.to_tensor(batch_action, dtype='int64')
        batch_reword = paddle.to_tensor(batch_reword, dtype='float32')
        batch_next_obs = paddle.to_tensor(batch_next_obs, dtype='float32')
        batch_done = paddle.to_tensor(batch_done, dtype='int64')

        return batch_obs, batch_action, batch_reword, batch_next_obs, batch_done

    def __len__(self):
        return len(self.buffer)
