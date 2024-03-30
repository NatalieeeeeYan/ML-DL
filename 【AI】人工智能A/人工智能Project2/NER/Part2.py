import numpy as np
import pickle
import check
from Common import TxtReader, TxtWriter
import sklearn_crfsuite

class CRF:

    def __get_features_from_word(self, sentence: list[str], index: int) -> dict:
        word = sentence[index]
        features = {
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'word.length()': len(word),
            'word.prefix-2': word[:2],
            'word.prefix-3': word[:3],
            'word.suffix-2': word[-2:],
            'word.suffix-3': word[-3:],
        }
        if index > 0:
            prev_word = sentence[index-1]
            features.update({
                'prev_word.lower()': prev_word.lower(),
                'prev_word.isupper()': prev_word.isupper(),
                'prev_word.istitle()': prev_word.istitle(),
                'prev_word.isdigit()': prev_word.isdigit(),
            })
        else:
            features['BOS'] = True

        if index < len(sentence)-1:
            next_word = sentence[index+1]
            features.update({
                'next_word.lower()': next_word.lower(),
                'next_word.isupper()': next_word.isupper(),
                'next_word.istitle()': next_word.istitle(),
                'next_word.isdigit()': next_word.isdigit(),
            })
        else:
            features['EOS'] = True

        return features
        
    def __init__(self) -> None:
        self.crf = sklearn_crfsuite.CRF()

    def train(self, group_of_states: list[list[str]], group_of_observations: list[list[str]]) -> None:
        """
        使用sklearn_crfsuite的crf.fit()和训练数据来训练模型
        :param group_of_states: 一组状态序列，每个状态序列是一个字符串列表
        :param group_of_observations: 一组观测序列，每个观测序列是一个字符串列表
        :return: None
        """
        X = [
                [
                    self.__get_features_from_word(sentence, i) 
                    for i 
                    in range(len(sentence))
                ]
                for sentence 
                in group_of_observations 
            ]
        y = group_of_states
        self.crf.fit(X, y)

    def test(self, group_of_observation: list[list[str]]) -> list[list[str]]:
        """
        使用sklearn_crfsuite的crf.predict()和测试数据来预测状态序列
        :param group_of_observation: 一组观测序列，每个观测序列是一个字符串列表
        :return: 一组状态序列
        """
        X = [
                [
                    self.__get_features_from_word(sentence, i) 
                    for i 
                    in range(len(sentence))
                ]
                for sentence 
                in group_of_observation 
            ]
        return self.crf.predict(X)

class ModelLoader:
    @staticmethod
    def load_from_train(train_data_path: str) -> CRF:
        """
        从训练集中加载模型
        :param train_data_path: 训练集路径
        :return: CRF模型
        """
        train_data = TxtReader(train_data_path)
        crf = CRF()
        crf.train(train_data.get_group_of_tags(), train_data.get_group_of_words())
        return crf    

    @staticmethod
    def save_model(language: str, crf: CRF, folder: str) -> None:
        """
        保存模型
        :param language: 语言
        :param crf: CRF模型
        :param folder: 保存文件夹
        :return: None
        """
        with open(f"{folder}/{language}.pkl", "wb") as f:
            pickle.dump(crf, f)

    @staticmethod
    def load_from_file(language: str, folder: str) -> CRF:
        """
        从文件中加载模型
        :param language: 语言
        :param path: 文件路径
        :return: CRF模型
        """
        with open(f"{folder}/{language}.pkl", "rb") as f:
            crf = pickle.load(f)
        return crf
    
    @staticmethod
    def test(test_data_path: str, crf: CRF, result_write_path: str) -> None:
        """
        测试模型
        :param test_data_path: 测试集路径
        :param crf: CRF模型
        :param result_write_path: 预测结果保存路径
        :return: None
        """
        txtWriter = TxtWriter(result_write_path)
        test_data = TxtReader(test_data_path)
        predicted_group_of_tags = crf.test(test_data.get_group_of_words())
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
    model_save_path = input(f"请输入模型保存文件夹(默认值为crf_model): ") or "crf_model"
    crf = ModelLoader.load_from_train(train_data_path)
    ModelLoader.save_model(language, crf, model_save_path)
else:
    model_load_path = input(f"请输入模型加载文件夹(默认值为crf_model): ") or "crf_model"
    crf = ModelLoader.load_from_file(language, model_load_path)

test_data_path = input(f"请输入测试集路径(默认值为{language}/validation.txt): ") or f"{language}/validation.txt"
result_write_path = input(f"请输入预测结果保存路径(默认值为exam/crf_{language}.txt): ") or f"exam/crf_{language}.txt"
ModelLoader.test(test_data_path, crf, result_write_path)
gold_path = input(f"请输入标准答案路径(默认值为{language}/validation.txt)") or f"{language}/validation.txt"
ModelLoader.check(language, gold_path, result_write_path)