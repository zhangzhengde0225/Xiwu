

import os, sys
from pathlib import Path
from FlagEmbedding import FlagModel
here = Path(__file__).parent.absolute()

class Embedding:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = FlagModel(self.model_path, 
                        # query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                        use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        return self._model
    
    def embed(self, text):
        return self.model.encode(text)
    
    def example(self):
        sentences_1 = ["你好", "样例数据-2"]
        sentences_2 = ["样例数据-3", "大聪妹"]
        
        embeddings_1 = self.model.encode(sentences_1)  # (2, 1025)
        embeddings_2 = self.model.encode(sentences_2)
        similarity = embeddings_1 @ embeddings_2.T
        print(similarity)

        # for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
        # corpus in retrieval task can still use encode() or encode_corpus(), since they don't need instruction
        queries = ['query_1', 'query_2']
        passages = ["样例文档-1", "样例文档-2"]
        q_embeddings = self.model.encode_queries(queries)
        p_embeddings = self.model.encode(passages)
        scores = q_embeddings @ p_embeddings.T
        print(scores)

if __name__ == "__main__":
    model_path = "/data/zzd/weights/baai/bge-large-zh-v1.5"
    embedding = Embedding(model_path=model_path)
    embedding.example()

    

   