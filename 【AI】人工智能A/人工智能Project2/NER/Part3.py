import torch
import torch.nn as nn
import torch.optim as optim
import check
from Common import TxtReader, TxtWriter
import pickle

# use GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1)

class BiLSTM_CRF(nn.Module):

    def log_sum_exp(self, mat):
        """
        对矩阵中的每一行计算log-sum-exp
        :param mat: torch.tensor, shape=(句子长度, 标签数量)
        :return: torch.tensor, shape=(句子长度, 1)
        """
        # max_score为了防止溢出，不影响结果。
        max_score = torch.max(mat, -1, keepdim=True)[0]
        result = max_score + \
            torch.log(torch.sum(torch.exp(mat - max_score.expand_as(mat)), -1, keepdim=True))
        return result

    def __init__(self, word_to_ix, tag_to_ix, embedding_dim, hidden_dim):
        """
        初始化BiLSTM_CRF模型.
        :param word_to_ix: dict, 单词到索引的映射.
        :param tag_to_ix: dict, 标签到索引的映射.
        :param embedding_dim: int, 词向量大小.
        :param hidden_dim: int, 隐藏层大小.
        """
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_to_ix = word_to_ix
        self.vocab_size = len(word_to_ix)
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # 嵌入层, LSTM层, 全连接层
        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim).to(device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True).to(device)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size).to(device)

        # 转移矩阵, t[i][j]表示从j转移到i的分数
        # 标签不可能转移到START_TAG, 也不可能从STOP_TAG转移出去
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)).to(device)
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        # LSTM的隐藏状态
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        随机初始化隐藏层参数.
        返回双元素元组, 分别表示隐藏状态(hidden)和细胞状态(cell).
        隐藏状态和细胞状态的维度都是(num_layers * num_directions, batch_size, hidden_size).
        num_directions = 2(双向LSTM), num_layers = 1(单层LSTM),
        batch_size = 1(每次输入一个句子), hidden_size = self.hidden_dim // 2(双向LSTM).
        """
        return (torch.randn(2, 1, self.hidden_dim // 2).to(device),
                torch.randn(2, 1, self.hidden_dim // 2).to(device))

    def _forward_alg(self, feats):
        """
        前向传播, 计算所有可能的路径的分数和.
        :param feats: torch.tensor, shape=(句子长度, 标签数量) 发射分数
        :return: torch.tensor, shape=(1, 1)
        """
        # 初始化forward_var为起始概率
        # 令START_TAG的概率为1(对数空间中为0), 其他标签的概率为0(对数空间中为-inf)
        forward_var = torch.full((1, self.tagset_size), -10000.).to(device)
        forward_var[0][self.tag_to_ix[START_TAG]] = 0.

        # 按顺序遍历句子中的每个词
        for feat in feats:
            feat = feat.view(self.tagset_size, 1).expand(
                self.tagset_size, self.tagset_size).to(device)
            # forward_var: 上一层的所有词转移到当前词的分数
            # feat: 发射分数 transitions: 转移分数
            forward_var = self.log_sum_exp(forward_var + feat + self.transitions).view(1, -1)
        # 最后一层的所有词转移到STOP_TAG的分数
        return self.log_sum_exp(forward_var + self.transitions[self.tag_to_ix[STOP_TAG]])

    def _get_lstm_features(self, sentence):
        """
        将句子中的每个词嵌入为词向量, 然后输入BiLSTM, 经过全连接层的转换, 得到输出的隐状态.
        :param sentence: list[int] 输入的句子, 每个元素是词的索引.
        :return: torch.tensor, shape=(句子长度, 标签数量)
        """
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1).to(device) # (sentence_length, batch_size, embedding_dim)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden) # (sentence_length, batch_size, hidden_dim)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim) # (sentence_length, hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out) # (sentence_length, tagset_size)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        """
        计算给定路径的分数.
        :param feats: torch.tensor, shape=(句子长度, 标签数量) 发射分数
        :param tags: list[int] 给定的路径，每个元素是标签的索引.
        :return: torch.tensor, shape=(1, 1)
        """
        score = torch.zeros(1).to(device)
        # 将START_TAG添加到路径的开头
        tags = torch.cat([
                torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device), 
                tags
            ])
        for i, feat in enumerate(feats):
            # 每个标签的发射分数 + 转移分数
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + \
                feat[tags[i + 1]]
        # 加上从最后一个标签转移到STOP_TAG的转移分数
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        """
        维特比算法, 在给定观测序列与发射分数情况下, 求最有可能的隐藏状态序列.
        :param feats: torch.tensor, shape=(句子长度, 标签数量) 发射分数
        :return: (number, List[Number]) 最大分数, 分数路径
        """
        
        # 初始化forward_var为起始概率
        # 令START_TAG的概率为1(对数空间中为0), 其他标签的概率为0(对数空间中为-inf).
        forward_var = torch.full((1, self.tagset_size), -10000.).to(device)
        forward_var[0][self.tag_to_ix[START_TAG]] = 0
        backtracks = []

        # 按序遍历每个单词
        for feat in feats:
            # max_score与max_track的shape都是(1, tagset_size)
            # max_score[0][i]表示前一层的所有标签转移到第i个标签的最大分数.
            # max_track[0][i]表示最大分数是由前一层的哪个标签转移过来的, 用于回溯. 
            max_score, max_track = torch.max(forward_var + self.transitions, dim=-1)
            # max_score: 路径分数和+转移分数  feat: 发射分数
            forward_var = (max_score + feat).view(1, -1)
            backtracks.append(max_track.tolist())

        # 最终转移到STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # 记录最大分数, 最大分数路径的最后一个标签
        last_tag = torch.argmax(terminal_var).item()
        path_score = terminal_var[0][last_tag]

        # 回溯, 找到最大分数路径
        best_path = [last_tag]
        best_tag_id = last_tag
        for max_track in reversed(backtracks):
            best_tag_id = max_track[best_tag_id]
            best_path.insert(0, best_tag_id)
        # 去除START_TAG
        del best_path[0]

        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        """
        计算负对数似然值.
        :param sentence: list[int] 输入的句子, 每个元素是词的索引.
        :param tags: list[int] 句子对应的标签, 每个元素是标签的索引.
        :return: torch.tensor, shape=(1, 1)
        """
        feats = self._get_lstm_features(sentence) # 发射分数 (sentence_length, tagset_size)
        forward_score = self._forward_alg(feats) # 所有路径的分数和
        gold_score = self._score_sentence(feats, tags) # 给定路径的分数
        return forward_score - gold_score

    def forward(self, sentence):
        """
        利用训练好的模型预测给定句子的标签.
        :param sentence: list[int] 输入的句子, 每个元素是词的索引.
        :return: (Number, list[Number]) 分数, 标签序列.
        """
        feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq
    
START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNKNOWN_WORD = "<UNK>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
EPOCH = 10

class ModelLoader:

    @staticmethod
    def prepare_sequence(seq, to_ix):
        # 将句子转换为索引序列
        # param: seq:["the", "ate"], to_ix:{"the": 0, "dog": 1, "ate": 2, "apple": 3, "cat": 4}
        # return: [0, 2]
        idxs = [to_ix.get(w, to_ix[UNKNOWN_WORD]) for w in seq]
        return torch.tensor(idxs, dtype=torch.long).to(device)

    @staticmethod
    def _train(model: BiLSTM_CRF, epoch: int, group_of_words: list[list[str]], group_of_tags: list[list[str]]) -> BiLSTM_CRF:
        """
        训练模型
        :param model: BiLSTM_CRF模型
        :param epoch: 迭代次数
        :return: 训练好的模型
        """
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
        for epoch in range(EPOCH): 
            import time
            start = time.time()
            for sentence, tags in zip(group_of_words, group_of_tags):
                model.zero_grad()

                sentence_in = ModelLoader.prepare_sequence(sentence, model.word_to_ix)
                targets = torch.tensor([model.tag_to_ix[t] for t in tags], dtype=torch.long).to(device)

                loss = model.neg_log_likelihood(sentence_in, targets)
                loss.backward()
                optimizer.step()
            end = time.time()
            print('Epoch: %d Loss: %.6f' % (epoch+1, loss.item()))
            print('Time: %.6f' % (end - start))
        return model

    @staticmethod
    def load_from_train(train_data_path: str) -> BiLSTM_CRF:
        """
        从训练集中加载模型
        :param train_data_path: 训练集路径
        :return: BiLSTM_CRF模型
        """
        train_data = TxtReader(train_data_path)
        word_to_ix = {word: i for i, word in enumerate(train_data.all_words)}
        word_to_ix[UNKNOWN_WORD] = len(word_to_ix)

        tag_to_ix = {tag: i for i, tag in enumerate(train_data.all_tags)}
        tag_to_ix[START_TAG] = len(tag_to_ix)
        tag_to_ix[STOP_TAG] = len(tag_to_ix)

        model = BiLSTM_CRF(word_to_ix, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM).to(device)
        model = ModelLoader._train(model, EPOCH, train_data.get_group_of_words(), train_data.get_group_of_tags())
        return model

    @staticmethod
    def save_model(language: str, model: BiLSTM_CRF, folder: str) -> None:
        """
        保存模型
        :param language: 语言
        :param model: 模型
        :param folder: 保存路径
        :return: None
        """
        with open(f"{folder}/{language}.pkl", "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_from_file(language: str, folder: str) -> BiLSTM_CRF:
        """
        从文件中加载模型
        :param language: 语言
        :param path: 文件路径
        :return: 模型
        """
        with open(f"{folder}/{language}.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    
    @staticmethod
    def test(test_data_path: str, model: BiLSTM_CRF, result_write_path: str) -> None:
        """
        测试模型
        :param test_data_path: 测试集路径
        :param model: BiLSTM_CRF模型
        :param result_write_path: 预测结果保存路径
        :return: None
        """

        txtWriter = TxtWriter(result_write_path)
        test_data = TxtReader(test_data_path)
        tag_ix_to_name = {i: tag for tag, i in model.tag_to_ix.items()}
        group_of_pred_tags = []
        with torch.no_grad():
            for words in test_data.get_group_of_words():
                _, pred_tag_ix = model(ModelLoader.prepare_sequence(words, model.word_to_ix))
                pred_tag = [tag_ix_to_name[i] for i in pred_tag_ix]
                group_of_pred_tags.append(pred_tag)

        txtWriter.write(group_of_pred_tags, test_data.get_group_of_words())

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
    model_save_path = input(f"请输入模型保存文件夹(默认值为bi_model): ") or "bi_model"
    model = ModelLoader.load_from_train(train_data_path)
    ModelLoader.save_model(language, model, model_save_path)
else:
    model_load_path = input(f"请输入模型加载文件夹(默认值为bi_model): ") or "bi_model"
    model = ModelLoader.load_from_file(language, model_load_path)

test_data_path = input(f"请输入测试集路径(默认值为{language}/validation.txt): ") or f"{language}/validation.txt"
result_write_path = input(f"请输入预测结果保存路径(默认值为exam/bi_{language}.txt): ") or f"exam/bi_{language}.txt"
ModelLoader.test(test_data_path, model, result_write_path)
gold_path = input(f"请输入标准答案路径(默认值为{language}/validation.txt)") or f"{language}/validation.txt"
ModelLoader.check(language, gold_path, result_write_path)