"""
使用文本Embedding算法+聚类算法，对知识问题进行聚类.

执行此程序：
在原来的XIWU格式的json文件中添加了一个clustering字段，里面包含了聚类的结果.
"""

import os, sys
from pathlib import Path
here = Path(__file__).parent.absolute()
from sklearn.cluster import KMeans, DBSCAN
from collections import OrderedDict
import numpy as np
from dataclasses import dataclass, field
import hai  # hepai
import json, ast

try:
    from xiwu.version import __version__
except:
    sys.path.append(str(here.parent.parent.parent))
    from xiwu.version import __version__

from xiwu.data.utils.flag_embedding import Embedding
from xiwu.apis import BaseQADatasetSaver
from xiwu.apis import HepAILLM

class Cluster:
    def __init__(self, model_path=None, **kwargs) -> None:
        self.method = kwargs.get('method', 'DBSCAN')
        self.eps = kwargs.get('eps', 0.45)
        self.min_samples = kwargs.get('min_samples', 5)
        self.embedding = Embedding(model_path)
        language = kwargs.get('language', 'en')
        llm = kwargs.get('llm', 'openai/gpt-3.5-turbo')
        self.llm = HepAILLM(language=language, model=llm)

    def kmeans(self, x):
        n_cluster = 10
        kmeans = KMeans(n_clusters=n_cluster, random_state=0)
        kmeans.fit(x)

        # 获取聚类标签
        labels = kmeans.labels_  # array: (500,)  # 每个样本所属的簇
        # 获取簇中心
        cluster_centers = kmeans.cluster_centers_  # array: (10, 1025)
        return labels, cluster_centers

    def dbscan(self, x, **kwargs):
        # 设置DBSCAN的参数
        epsilon = kwargs.get('eps', self.eps) # 邻域大小
        min_samples = kwargs.get('min_samples', self.min_samples)  # 最小样本数

        # 初始化DBSCAN
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        # 拟合数据
        dbscan.fit(x)

        # 获取聚类标签
        labels = dbscan.labels_

        # 标签中的-1表示噪声点
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 获取簇的个数

        # 你现在可以使用labels来了解每个数据点属于哪个簇，以及哪些点被认为是噪声
        return labels, n_clusters


    def run(self, data_path):
        data_obj = BaseQADatasetSaver(data_path)
        entities = data_obj.data['entities']

        sentences = list()
        for i, entity in enumerate(entities):
            question = entity['question']
            answer = entity['answer']
            # print(f'Embedding {i:0>3}th question: {question}')
            text = f'{question}\n{answer}'
            sentences.append(text)
        embeddings = self.embedding.embed(sentences)

        # self.kmeans(embeddings)
        labels, n_clusters = self.dbscan(embeddings)  # -1代表离散点

        # 根据聚类的结果，对同一类获取主题，并获取主题涉及的概念，返回dict
        cluster_result = self.cluster_results(labels, n_clusters, entities)

        # 保存聚类结果
        data = data_obj.data
        data['clustering'] = cluster_result
        data_obj._save_data2file(data)

    def texts2topics(self, texts):
        """将文本转换为1个主题, 用GPT-3.5"""
        sys_prompt = "Your task is to summarize the text and summarize the topics they describe from the input information"
        exmple = '{"The topic of the above input is:", <TOPIC>}'
        prompt = f"""
Please summarize one topics they describe from the input information.
Only output one topic. Please by concise and specific.
Output should be JSON format. For example: {exmple}

Input:
```{texts}```

"""
        res = self.llm.generate(
            prompt=prompt,
            sys_prompt=sys_prompt,
            return_format='json_object',
    
        )
        res = self.llm.parse_json(res)
        assert len(res) == 1, f'len(res)={len(res)}'
        res = list(res.values())[0]
        return res
    
    def text2concepts(self, texts):
        """
        从文本中提取涉及到的概念
        """
        sys_prompt = "Your task is to extract keywords from the input text"
        exmaple = """
- <CONCEPT1>
- <CONCEPT2>
"""
        prompt = f"""
Please extract keywords from the input text delimited by triple backticks.
Output should be a list of keywords. For example: {exmaple}
Please by concise and specific.

Input:
```{texts}```

The keywords of above input are:
"""
        res = self.llm.generate(
            prompt=prompt,
            sys_prompt=sys_prompt,
        )
        # 转换为list
        # 使用字符串的split方法按照换行符分割文本
        list_items = res.split('\n')

        # 清理每一项，移除前面的'- '前缀
        list_items = [item.strip('- ').strip() for item in list_items]
        return list_items

    def cluster_results(self, labels, n_clusters, entities):
        """聚类结果写到json文件里"""

        topics = list()
        unique_labels = [int(x) for x in set(labels)]
        statics = dict()
        assert len(labels) == len(entities), f'len(labels) {len(labels)} should be equal to len(entities) {len(entities)}'
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            if label == -1:
                statics[label] = {
                    'topic': 'misc',
                    'n_entities': len(indices),
                }
                continue
            entities_in_cluster = [entities[i] for i in indices]

            questions = [x['question'] for x in entities_in_cluster]
            questions = '\n'.join(questions)
            topic = self.texts2topics(questions)
            topics.append(topic)

            # concepts = []
            # for entity in entities_in_cluster:
            #     qa = f"{entity['question']}\n{entity['answer']}"
            #     concepts_ = self.text2concepts(qa)
            #     concepts.extend(concepts_)
            # concepts = list(set(concepts))

            statics[label] = {
                'topic': topic,
                'n_entities': len(entities_in_cluster),
                # "concepts": concepts,
            }

        int_labels = [int(x) for x in list(labels)]
        int_unique_labels = [int(x) for x in list(unique_labels)]
        clustering = dict()
        clustering['method'] = self.method
        clustering['params'] = {
            'eps': self.eps,
            'min_samples': self.min_samples,
        }
        clustering['n_clusters'] = int(n_clusters)
        clustering['labels'] = int_labels
        clustering['unique_labels'] = int_unique_labels
        clustering['topics'] = topics
        clustering['statics'] = statics


        return clustering


@dataclass
class Args:
    model_path: str = "/data/zzd/weights/baai/bge-large-zh-v1.5"
    method: str = 'FlagEmbedding and DBSCAN'
    eps: float = 0.45
    min_samples: int = 5
    data_path: str = f'{here.parent}/seed_fission/generated_data/particle_physics_QA_datasets.json'
    language: str = 'en'
    llm: str = 'openai/gpt-4'  # For summary topics

if __name__ == "__main__":
    args = hai.parse_args(Args)
    c = Cluster(
        model_path=args.model_path,
        method=args.method,
        eps=args.eps,
        min_samples=args.min_samples,
        language=args.language,
        llm=args.llm,
        )
    c.run(data_path=args.data_path)