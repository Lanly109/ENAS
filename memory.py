import torch
import numpy as np

from sumtree import SumTree

class ReplayMemory(object):

    def __init__(
            self,
            layer_limit,
            capacity,
            device
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.tree = SumTree(capacity)
        self.belta = 0.4
        self.belta_increment_per_sampling = 0.001
        self.__size = 0
        self.epsilon = 0.01
        self.alpha = 0.6
        self.abs_err_upper = 1.0
        self.__pos = 0
        self.__m_probs = torch.zeros((capacity, layer_limit * 2 - 1), dtype=torch.float).to(self.__device)
        self.__m_actions = torch.zeros((capacity, layer_limit * 2 - 1), dtype=torch.int8).to(self.__device)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.float).to(self.__device)

    def push(
            self,
            prob,
            action,
            reward,
    ) -> None:
        max_p = self.tree.max_p
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p)
        self.__m_probs[self.__pos] = prob
        self.__m_actions[self.__pos] = torch.from_numpy(action)
        self.__m_rewards[self.__pos, 0] = reward
       
        self.__pos = (self.__pos + 1) 
        self.__size = max(self.__size, self.__pos)
        self.__pos = self.__pos % self.__capacity

    def sample(self, batch_size: int):
        pri_step = self.tree.total_p / batch_size
        self.belta = np.min([1, self.belta + self.belta_increment_per_sampling])
        min_p = self.tree.min_p / self.tree.total_p
        min_p = self.epsilon if min_p <= self.epsilon else min_p
        v = [np.random.uniform(i * pri_step, (i + 1) * pri_step) for i in range(batch_size)]
        indices, ISweight = self.tree.get_leaf(v)
        ISweight = [[np.power((i + self.epsilon) / min_p, - self.belta)] for i in ISweight]
        b_prob = self.__m_probs[indices].to(self.__device)
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        ISweight = torch.Tensor(np.array(ISweight)).to(self.__device).float()
        return indices, b_prob, b_action, b_reward, ISweight

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        # upper bound 1
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti + self.__capacity - 1, p)

    def __len__(self) -> int:
        return self.__size
