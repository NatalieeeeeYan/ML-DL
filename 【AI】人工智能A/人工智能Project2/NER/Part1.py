import numpy as np
import pickle
import check
from Common import TxtReader, TxtWriter

class HMM:
    def __init__(self, all_states: list[str], all_observations: list[str]) -> None:
        self.states = all_states
        self.observations = all_observations
        self.start_prob = None # |S| * 1
        self.transition_prob = None # |S| * |S|
        self.emission_prob = None # |S| * |O|

    def train(self, group_of_states: list[list[str]], group_of_observations: list[list[str]]) -> None:
        """
        训练模型，HMM的start_prob, transition_prob, emission_prob都会被赋值
        :param group_of_states: 一组状态序列，每个状态序列是一个字符串列表
        :param group_of_observations: 一组观测序列，每个观测序列是一个字符串列表
        :return: None
        """
        # start_prob
        first_state_of_each_group = [group[0] for group in group_of_states]
        start_count = [first_state_of_each_group.count(state) for state in self.states]
        self.start_prob = np.array(start_count)
        self.start_prob[self.start_prob == 0] = 5e-2 # 防止出现0，导致后面计算log时出现-inf
        self.start_prob = self.start_prob / np.sum(self.start_prob)
        self.start_prob = np.log2(self.start_prob)

        # transition_prob
        transition_count = np.zeros((len(self.states), len(self.states)))
        for group in group_of_states:
            for i in range(len(group) - 1):
                from_state, to_state = group[i], group[i + 1]
                transition_count[self.states.index(from_state)][self.states.index(to_state)] += 1
        transition_count[transition_count == 0] = 5e-2 # 防止出现0，导致后面计算log时出现-inf
        self.transition_prob = transition_count / np.sum(transition_count, axis=1, keepdims=True)
        self.transition_prob = np.log2(self.transition_prob)

        # emission_prob
        emission_count = np.zeros((len(self.states), len(self.observations)))
        flat_states = [state for group in group_of_states for state in group]
        flat_observations = [observation for group in group_of_observations for observation in group]
        for s, o in zip(flat_states, flat_observations):
            emission_count[self.states.index(s)][self.observations.index(o)] += 1
        emission_count[emission_count == 0] = 5e-2 # 防止出现0，导致后面计算log时出现-inf
        self.emission_prob = emission_count / np.sum(emission_count, axis=1, keepdims=True)
        self.emission_prob = np.log2(self.emission_prob)

    def __viterbi(self, observations: list[str]) -> list[str]:
        """
        维特比算法，用于预测一组观测序列对应的状态序列
        :param observations: 一组观测序列
        :return: 一组状态序列
        """

        def __get_emission_prob(observation: str) -> np.ndarray: # |S| * 1
            """
            获得emission_prob中对应的列, 如果observation不在observations中，则返回均匀分布
            :param observation: 观测序列中的一个观测
            :return: emission_prob中对应的列
            """
            if observation in self.observations:
                return self.emission_prob[:, self.observations.index(observation)]
            else:
                return np.ones(len(self.states)) * np.log(1.0 / len(self.states))
        
        # Initialize
        deltas = np.zeros((len(self.states), len(observations)))
        tracks = np.zeros((len(self.states), len(observations)), dtype=int)
        deltas[:, 0] = self.start_prob + __get_emission_prob(observations[0])
        
        # Recursion
        for t in range(1, len(observations)):
            delta = deltas[:, t-1] + self.transition_prob.T
            tracks[:, t] = np.argmax(delta, axis=1)
            deltas[:, t] = np.max(delta, axis=1) + __get_emission_prob(observations[t])
            
        # Backtracking
        path = [np.argmax(deltas[:, -1])]
        for t in range(len(observations)-2, -1, -1):
            path.insert(0, tracks[path[0], t+1])
        return [self.states[p] for p in path]

    def test(self, group_of_observation: list[list[str]]) -> list[list[str]]:
        """
        预测一组观测序列对应的状态序列
        :param group_of_observation: 一组观测序列，每个观测序列是一个字符串列表
        :return: 一组状态序列，每个状态序列是一个字符串列表
        """
        return [self.__viterbi(observation) for observation in group_of_observation]
        
class ModelLoader:
    @staticmethod
    def load_from_train(train_data_path: str) -> HMM:
        """
        从训练集中加载模型
        :param train_data_path: 训练集路径
        :return: HMM模型
        """
        train_data = TxtReader(train_data_path)
        hmm = HMM(train_data.get_all_tags(), train_data.get_all_words())
        hmm.train(train_data.get_group_of_tags(), train_data.get_group_of_words())
        return hmm    

    @staticmethod
    def save_model(language: str, hmm: HMM, folder: str) -> None:
        """
        保存模型
        :param language: 语言
        :param hmm: HMM模型
        :param path: 保存路径
        :return: None
        """
        with open(f"{folder}/{language}.pkl", "wb") as f:
            pickle.dump(hmm, f)

    @staticmethod
    def load_from_file(language: str, folder: str) -> HMM:
        """
        从文件中加载模型
        :param language: 语言
        :param path: 文件路径
        :return: HMM模型
        """
        with open(f"{folder}/{language}.pkl", "rb") as f:
            hmm = pickle.load(f)
        return hmm
    
    @staticmethod
    def test(test_data_path: str, hmm: HMM, result_write_path: str) -> None:
        """
        测试模型
        :param test_data_path: 测试集路径
        :param hmm: HMM模型
        :param result_write_path: 预测结果保存路径
        :return: None
        """
        txtWriter = TxtWriter(result_write_path)
        test_data = TxtReader(test_data_path)
        predicted_group_of_tags = hmm.test(test_data.get_group_of_words())
        txtWriter.write(predicted_group_of_tags, test_data.get_group_of_words())

    @staticmethod
    def check(language: str, gold_path: str, my_path: str) -> None:
        """
        检查预测结果
        :param language: 语言
        :param gold_path: 标准答案路径
        :param my_path: 预测结果路径
        :return: None
        """
        check.check(language, gold_path, my_path)
        

language = input("请输入语言(1-English/2-Chinese): ")
assert language in ["1", "2"], "输入有误[1或2]，请重新输入。"
language = "English" if language == "1" else "Chinese"

load_method = input("请输入加载模型的方式(1-训练集/2-训练好的模型): ")
assert load_method in ["1", "2"], "输入有误[1或2]，请重新输入。"
if load_method == '1':
    train_data_path = input(f"请输入训练集路径(默认值为{language}/train.txt): ") or f"{language}/train.txt"
    model_save_path = input(f"请输入模型保存文件夹(默认值为hmm_model): ") or "hmm_model"
    hmm = ModelLoader.load_from_train(train_data_path)
    ModelLoader.save_model(language, hmm, model_save_path)
else:
    model_load_path = input(f"请输入模型加载文件夹(默认值为hmm_model): ") or "hmm_model"
    hmm = ModelLoader.load_from_file(language, model_load_path)

test_data_path = input(f"请输入测试集路径(默认值为{language}/validation.txt): ") or f"{language}/validation.txt"
result_write_path = input(f"请输入预测结果保存路径(默认值为exam/hmm_{language}.txt): ") or f"exam/hmm_{language}.txt"
ModelLoader.test(test_data_path, hmm, result_write_path)
gold_path = input(f"请输入标准答案路径(默认值为{language}/validation.txt)") or f"{language}/validation.txt"
ModelLoader.check(language, gold_path, result_write_path)