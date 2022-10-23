

class ContextClass:
    def __init__(self):
        self.collected_rewards = []
        self.binary_features = []
        self.learners = []
        self.current_split = []
        self.split_list = [[-1, -1], [0, -1], [1, -1], [0, 0], [0, 1], [1, 0], [1, 1]]
        self.pending_list = []


    def update(self, reward):
        self.collected_rewards.append(reward)


    def split(self, check = True, l_reward = -1, r_reward = -1):
        if len(self.current_split) == 0 and l_reward == -1 and r_reward == -1:
            self.current_split.append(split_list[0])
        elif current_split[0] == [-1, -1]:
            self.current_split.pop(0)
            self.split_list.pop(0)
            self.current_split.append(self.split_list[:2])
        elif self.current_split[0] == [0, -1] and self.current_split[1] == [1, -1]:
            if check:
                if l_reward > r_reward:
                    self.pending_list.append(self.current_split[1])
                    self.split_list.pop(0)
                    self.split_list.pop(0)
                    self.current_split = self.split_list[:2]
                else:
                    self.pending_list.append(self.current_split[0])
                    self.current_split = self.split_list[2:4]
        elif any(-1 not in sublist for sublist in self.split_list):
            if check:
                if l_reward > r_reward:
                    self.pending_list.append(self.current_split[1])
                    self.current_split = self.split_list[:2]
                else:
                    self.pending_list.append(self.current_split[0])
                    self.current_split = self.split_list[2:4]
            elif len(self.pending_list) > 0:
                if self.pending_list[0] == [0, -1] or self.pending_list[0][1] == 0:
                    self.pending_list.pop(0)
                    self.current_split = self.split_list[:2]
                if self.pending_list[0] == [1, -1] or self.pending_list[0][1] == 1:
                    self.pending_list.pop(0)
                    self.current_split = self.split_list[2:4]
                self.split()







    def evaluate_split_condition(self):

        return check, l_reward, r_reward

    def generate_context_bandit(self):


    def run_aggregate_context(self):
        return reward_ts, reward_ucb