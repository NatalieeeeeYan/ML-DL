
class TxtReader:
    def __init__(self, path: str) -> None:
        self.path = path
        self.group_of_words, self.group_of_tags = self.__get_data()
        self.all_words = sorted(set(word for sentence in self.group_of_words for word in sentence))
        self.all_tags = sorted(set(tag for tag_list in self.group_of_tags for tag in tag_list))

    def __split_lines(self, lines: list[str]) -> tuple[list[str], list[str]]:
        """
        将一行中的文字与标签分开
        例如: ['我 O\n', '爱 O  \n', '你 O\n', '中 B-LOC', '国 I-LOC', '。 O']
        将被分成: ['我', '爱', '你', '中', '国', '。'], ['O', 'O', 'O', 'B-LOC', 'I-LOC', 'O']
        """
        words, tags = [], []
        for line in lines:
            ls = line.strip().split()
            if len(ls) != 2: continue
            words.append(ls[0])
            tags.append(ls[1])
        return words, tags
    
    def __get_data(self) -> tuple[list[list[str]], list[list[str]]]:
        """
        从txt中读取数据，并将文字与标签分开返回
        return: (list[list[str]], list[list[str]]), (文字列表, 标签列表)
                列表中的每个元素为一句话，每句话中的每个元素为一个字或标签
        """
        with open(self.path, 'r') as f: 
            lines = f.readlines()
        if lines[-1] != '\n': 
            lines.append('\n')
        sentences, tags = [], []
        while lines.count('\n') > 0:
            index = lines.index('\n')
            w, t = self.__split_lines(lines[:index])
            if w:
                sentences.append(w)
                tags.append(t)
            lines = lines[index + 1:]
        return sentences, tags

    def get_group_of_words(self) -> list[list[str]]:
        """
        获得单词列表，列表中的每个元素为一句话，每句话中的每个元素为一个字
        [[word1, word2, ...], [word1, word2, ...], ...]
        """
        return self.group_of_words
    
    def get_group_of_tags(self) -> list[list[str]]:
        """
        获得标签列表，列表中的每个元素为一句话，每句话中的每个元素为一个标签
        [[tag1, tag2, ...], [tag1, tag2, ...], ...]
        """
        return self.group_of_tags
    
    def get_all_words(self) -> list[str]:
        """
        获得所有出现过的单词(已经过打平、去重、排序)
        [aWord, bWord, ...]
        """
        return self.all_words
    
    def get_all_tags(self) -> list[str]:
        """
        获得所有出现过的标签(已经过打平、去重、排序)
        [aTag, bTag, ...]
        """
        return self.all_tags

class TxtWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        # test self.path whether is a valid path and can be written
        with open(self.path, 'w') as f:
            pass

    def write(self, group_of_states: list[list[str]], group_of_observations: list[list[str]]) -> None:
        """
        将预测结果写入文件
        :param group_of_states: 一组状态序列，每个状态序列是一个字符串列表
        :param group_of_observations: 一组观测序列，每个观测序列是一个字符串列表
        :return: None
        """
        with open(self.path, 'w') as f:
            for states, observations in zip(group_of_states, group_of_observations):
                for state, observation in zip(states, observations):
                    f.write(f'{observation} {state}\n')
                f.write('\n')

