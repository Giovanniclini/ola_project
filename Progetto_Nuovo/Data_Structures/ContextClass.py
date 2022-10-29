import numpy as np


class ContextClass:
    def __init__(self):
        self.collected_rewards = []
        self.binary_features = []
        self.learners = []
        self.current_split = []
        self.split_list = [[-1, -1], [0, -1], [1, -1], [0, 0], [0, 1], [1, 0], [1, 1]]
        self.pending_list = []
        self.pending_list_lower_bounds = []
        self.pending_list_prob = []
        self.father_lower_bound = []
        self.reward_index = -1

    def update(self, reward):
        self.collected_rewards.append(reward)

    def split(self, check=True, l_reward=-1, r_reward=-1):
        if len(self.current_split) == 0 and l_reward == -1 and r_reward == -1:
            self.current_split.append(self.split_list[0])
        elif self.current_split[0] == [-1, -1]:
            self.current_split.pop(0)
            self.split_list.pop(0)
            self.current_split.append(self.split_list[0])
            self.current_split.append(self.split_list[1])
        elif self.current_split[0] == [0, -1] and self.current_split[1] == [1, -1]:
            if check:
                if l_reward > r_reward:
                    self.reward_index = 0
                    self.pending_list.append(self.current_split[1])
                    self.split_list.pop(0)
                    self.split_list.pop(0)
                    self.current_split = self.split_list[:2]
                else:
                    self.reward_index = 1
                    self.pending_list.append(self.current_split[0])
                    self.current_split = self.split_list[2:4]
        elif any(-1 not in sublist for sublist in self.split_list):
            if check:
                if l_reward > r_reward:
                    self.reward_index = 0
                    self.pending_list.append(self.current_split[1])
                    self.current_split = self.split_list[:2]
                else:
                    self.reward_index = 1
                    self.pending_list.append(self.current_split[0])
                    self.current_split = self.split_list[2:4]
            elif len(self.pending_list) > 0:
                self.reward_index = -1
                if self.pending_list[0] == [0, -1] or self.pending_list[0][1] == 0:
                    self.pending_list.pop(0)
                    self.current_split = self.split_list[:2]
                if self.pending_list[0] == [1, -1] or self.pending_list[0][1] == 1:
                    self.pending_list.pop(0)
                    self.current_split = self.split_list[2:4]

    def evaluate_split_condition(self, rewards0, rewards1, time):
        check = False
        l_lowerbound = self.lower_bound(rewards0, 5, 14)
        r_lowerbound = self.lower_bound(rewards1, 5, 14)
        p = self.assign_prob_context_occur(time)
        if p * (l_lowerbound + r_lowerbound) >= self.father_lower_bound:
            check = True
        return check, l_lowerbound, r_lowerbound

    def assign_father_lower_bound(self, rewards):
        self.father_lower_bound = self.lower_bound(rewards, 5, 14)

    def lower_bound(self, rewards, confidence, cardinality):
        return np.mean(rewards) - np.sqrt(-np.log10(confidence)/(cardinality * 2))

    def assign_prob_context_occur(self, time):
        return 1 / (2 ** (time / 14)) * 100

