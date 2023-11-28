
import os, sys
import ast
import numpy as np
import random
import re
import json
from json import JSONDecodeError

from xiwu.apis.xiwu_api import HepAILLM

class TopicBot(HepAILLM):

    def __init__(self, system_prompt=None, **kwargs):
        system_prompt = f"You are a robot that determines which topic is more interesting. You need to select the most different  one"
        super().__init__(system_prompt)

        self._topics = []
        # self._simularity_matrix = np.zeros()  # 自相似矩阵，用于存储问题之间的相似度

    @property
    def topics(self):
        return self._topics
    
    @property
    def simularity_matrix(self):
        return self._simularity_matrix
    
    def parse_topics(self, topics):
        """
        解析topics,
        :param new_topics: str, "{'Q1': '<ONE QUESTION>', 'Q2': '<ONE QUESTION>'}"
        """
        try:
            new_topics = ast.literal_eval(topics)  # str2dict
        except ValueError as e:
            # match the texts in the ```json\nxxx\n``` format
            # pattern = re.compile(r'```json\n(.*)\n```', re.S)
            # new_topics = re.findall(pattern, topics)
            new_topics = topics.split('\n\n')[0]
            new_topics = ast.literal_eval(new_topics)  # str2dict
        except Exception as e:
            topics = topics.strip('```json')
            topics = topics.strip('```')
            # 去掉最开始的回车符直到出现第一个{
            topics = topics.lstrip('\n')
            topics = topics.rstrip('\n')
            try:
                new_topics = json.loads(topics)
            except JSONDecodeError as e:
                try:
                    new_topics = ast.literal_eval(topics)  # str2dict
                except Exception as e:
                    with open('error_toipc.txt', 'w') as f:
                        f.write(topics)
                    error = {'error': f'Parse topic {topics} Error: \n{e}'}
                    raise Exception(error)
            # raise Exception(f"Parse topic {topics} Error: \n{e}")
        topics_list = [v for k, v in new_topics.items() if 'Q' in k]
        return topics_list
    
    def prompt_engineering(self, topics):
        tmp_candidate_topics = ''
        discussed_topics = ''
        example = "{'Most interesting topic': 'xxx'}"
        example = "['<TOPIC1>': {'intresting score': 5}, '<TOPIC2>', 'intresting score': 8]"
        example = "{'The most differenct topic is': '<TOPIC>'}"
        example = "{'Compare 1': {'candidate': '<CANDIDATE TOPIC1>', 'discussed': '<DISCUSSED TOPIC1>', 'Simularity Socre': <SCORE>}, 'Compare 2': {'candidate': '<CANDIDATE TOPIC2>', 'discussed': '<DISCUSSED TOPIC2>', 'Simularity Socre': <SCORE>}, 'Compare 3': ...}"
        prompt = f"""
Please choose the most differnet topic from the candidate topics in the triple backticks from the discussed topics in the triple triple dashes.
Note that the selected topic must be exactly the same as one of the candidate topics.
Replace the content of angle brackets in the example with a real topic in candidate topics.
Provide it in JSON format. For example: {example}.
{self.prompt_lang}

candidate topics:
```{tmp_candidate_topics}```

discussed topics:
---{discussed_topics}---
        """
        print('Topic Prompt: ', prompt)
        # print('candidate_topics: ', self.candidate_topics)

        print('TopicGPT:\n', end='')
        selected_topic = self.topic_gpt(prompt, sys_prompt=None)
        # 解析
        selected_topic = list(ast.literal_eval(selected_topic).values())
        assert len(selected_topic) == 1, f'selected_topic should be one {selected_topic}'
        selected_topic = selected_topic[0]
    
    def cal_differ(self, str1, str2):
        pass

    def diff_of_str1_and_str2(self, str1, str2):
        import difflib
        # str1 = 'hello world'
        # str2 = 'hello python'
        s = difflib.SequenceMatcher(None, str1, str2)
        # print(s.ratio())
        return s.ratio()
    
    def _duplicat_remove(self, topics, exist_topics=None, threshold=0.9, **kwargs):
        """去重"""
        need_self_duplicat_remove = kwargs.get('need_self_duplicat_remove', True)

        # 完全相同的去重
        topics = list(set(topics)) 
        # 自相似去重
        if need_self_duplicat_remove:
            new_topics = []
            copy_topics = topics.copy()
            for topic in reversed(copy_topics):
                simularities = [self.diff_of_str1_and_str2(topic, topic2) for topic2 in copy_topics if topic2 != topic]
                print(f"自相关: {[f'{x:.2f}' for x in simularities]} Topic: {topic}")
                if max(simularities) < threshold:
                    new_topics.append(topic)
            topics = reversed(new_topics)

        # 与已有的去重
        if exist_topics is not None:
            new_topics = []
            for topic in topics:
                simularities = [self.diff_of_str1_and_str2(topic, exist_topic) for exist_topic in exist_topics]
                print(f"互相关: {[f'{x:.2f}' for x in simularities]} Topic: {topic}")
                simularities = simularities if len(simularities) > 0 else [0]
                if max(simularities) < threshold:
                    new_topics.append(topic)
            topics = new_topics
        return topics
    
    def _safe_sample(self, a_list, num=1):
        if len(a_list) > num:
            new = random.sample(a_list, num).copy()
        else:
            new = a_list.copy()
        return new

    
    def random_select(self, new_topics, **kwargs):
        """随机选择一个topic"""
        candidate_topics = kwargs.get('candidate_topics', None)
        assert candidate_topics is not None, f'candidate_topics is None'
        num = len(new_topics) * 2
        candi_t = self._safe_sample(candidate_topics, num)
        tmp_candi_topics = candi_t + new_topics
        if len(tmp_candi_topics) == 0:  # 如果产生new_topics为空，会导致总tmp_candi_topics也为空，此时从总的候选中随机跳出5个补充
            tmp_candi_topics = self._safe_sample(candidate_topics, 5)
        print(f'tmp_candi_topics: {candi_t} + {new_topics}')
        selected_topic = random.sample(tmp_candi_topics, 1)[0]
        return selected_topic

    
    def screen_topics(self, new_topics, **kwargs):
        """筛选掉相似度高的"""
        # exist_topics = kwargs.get('exist_topics', None)
        
        save_callback = kwargs.get('save_callback', None)
        # 解析
        new_topics = self.parse_topics(new_topics)  # a list
        # 去重: 完全相同的去重，自相似去重，与已有的去重
        new_topics = self._duplicat_remove(new_topics, **kwargs)
        
        # 随机选择一个topic
        selected_topic = self.random_select(new_topics, **kwargs)
        
        # 回调函数保存未使用的新topic
        assert save_callback is None or callable(save_callback), f'save_callback should be callable {save_callback}'
        if save_callback is not None:
            for t in new_topics:
                if t != selected_topic:
                    save_callback(t)
        return selected_topic

