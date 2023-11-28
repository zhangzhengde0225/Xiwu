"""
使用种子裂变技术对1个种子问题进行扩展，裂变n次得到2^(n+1)-1个问题和答案
裂变10次可累计获得2047个问答对
"""
import os, sys
import ast
from pathlib import Path
import hai
import damei as dm
import random
import re
import numpy as np
from dataclasses import dataclass, field

here = Path(__file__).parent

try:
    from xiwu import __version__
except:
    sys.path.append(str(here.parent.parent.parent))
    from xiwu import __version__
    
from xiwu.apis import ExpertBot, NewbeeBot, TopicBot, BaseQADatasetSaver
logger = dm.get_logger("seed_fission.py")

class SeedFission(BaseQADatasetSaver):
    def __init__(self, domain, language='zh', **kwargs):
        super().__init__(**kwargs)
    
        self.newbee_bot = NewbeeBot(domain, language=language)
        self.expert_bot = ExpertBot(domain, language=language)
        self.topic_bot = TopicBot(language=language)
        self._candidate_topics = self.data['metadata'].get('candidate_topics', list())
    
    @property
    def num_enrities(self):
        return len(self.data['entities'])

    @property
    def exist_questions(self):
        return [qa['question'] for qa in self.data['entities']]
    
    @property
    def candidate_topics(self):
        self._candidate_topics = self.data['metadata'].get('candidate_topics', list())
        return self._candidate_topics

    def add_candidate_topic(self, topic):
        self._candidate_topics.append(topic)
        self._candidate_topics = list(set(self._candidate_topics))
        self.data['metadata']['candidate_topics'] = self._candidate_topics

    def cal_simularity_by_topic_gpt(self, question1, question2):
        """计算两个问题的相似度"""
        pass

    def determine_most_interesting_topic(self, new_topics):
        """prompt for determining the most interesting topic"""
        discussed_topics = '---\n---'.join(self.exist_questions)
        # candidate_topics = '\n'.join(self.candidate_topics)
        if len(self.candidate_topics) >= 2:
            sample_cts = random.sample(self.candidate_topics, 2).copy()  # 随机采样2个
        else:
            sample_cts = self.candidate_topics.copy()
        tmp_candidate_topics_list = sample_cts + new_topics
        tmp_candidate_topics = '```\n```'.join(tmp_candidate_topics_list)
        print('tmp_candidate_topics: ', tmp_candidate_topics)
        
        use_topic_gpt = False
        if use_topic_gpt:
            pass
        else:
            selected_topic = random.sample(tmp_candidate_topics_list, 1)[0]
        assert selected_topic in tmp_candidate_topics_list, f'selected_topic {selected_topic} should be in candidate_topics {tmp_candidate_topics_list}'
            
        
        self.add_candidate_topic()
        
        return selected_topic
    
    def run(self, seed, max_entities=2000, **kwargs):
        num_to_gen = kwargs.get('num_to_gen', None)
        count = 0
        # cold_start = True
        while True:
            if self.num_enrities >= max_entities:
                logger.info(f"Max entities {max_entities} reached, stop.")
                break
            elif num_to_gen is not None and count >= num_to_gen:
                logger.info(f"Num to gen {num_to_gen} reached, stop.")
                break
            n = self.num_enrities
            if n == 0:
                input = seed
            else:
                input = self.data['entities'][-1]['answer']
            questions = self.newbee_bot.input2questions(input, n=n, **kwargs)
            # questions = self.newbee_bot.input2questions(input, n=n, model='openai/gpt-3.5-turbo')
            question = self.topic_bot.screen_topics(
                questions,
                n=n,
                exist_topics=self.exist_questions,
                candidate_topics=self.candidate_topics,
                save_callback=self.add_candidate_topic,
                **kwargs
                )
            # question = self.determine_most_interesting_topic(questions)
            answer = self.expert_bot.question2answer(question, n=n, **kwargs)

            qa_pair = dict()
            qa_pair['question'] = question
            qa_pair['answer'] = answer
            entity = self.qa_pair2entity(qa_pair, **kwargs)
            self._add_one_entity_and_save(entity, duplicate_remove=True)
            count += 1
    
    def qa_pair2entity(self, qa_pair, **kwargs):
        entity_info = kwargs.get('entity_info', None)
        if entity_info is None:
            entity_info = {
                'category': None,
                "source": None,
                "labeler": None,
            }
        qa_pair.update(entity_info)
        return qa_pair
    
def main(args):
    
    kwargs = args.__dict__
    domain = kwargs.get('domain', None)
    seed = kwargs.get('seed', None)
    seed = seed if seed else domain  # 如果seed为None，则使用domain作为seed
    num_to_gen = kwargs.get('num_to_gen')
    max_entities = kwargs.get('max_entities')
    output_dir = kwargs.get('output_dir')
    model = kwargs.get('model')
    version = args.version
    metadata = {
        'description': args.description,
        'version': version,
    }
    entity_info = {
        'category': domain,
        "source": args.model,
        "labeler": args.author,
    }

    output_file = f'{"_".join(domain.split()).lower()}_QA_datasets.json'
    output_file = f'{output_dir}/{output_file}'
    sf = SeedFission(
        domain,
        language=args.language, 
        version=args.version, 
        metadata=metadata,
        output_file=output_file)
    sf.run(seed, num_to_gen=num_to_gen, max_entities=max_entities, entity_info=entity_info, model=model)

@dataclass
class Args:
    domain: str = field(default='Particle Physics', metadata={'help': 'domain to generate QA dataset'})
    seed: str = field(default=None, metadata={'help': 'seed to generate QA dataset, if None, use domain'})
    num_to_gen: int = field(default=None, metadata={'help': 'number of QA pairs to generate in one run'})
    max_entities: int = field(default=500, metadata={'help': 'max number of entities to generate for all runs'})
    output_dir: str = field(default=f'{here}/generated_data', metadata={'help': 'output file'})
    model: str = field(default='openai/gpt-4', metadata={'help': 'model to use'})
    language: str = field(default='en', metadata={'help': 'language to use, must be one of [en, zh]'})
    description: str = field(default='The QA dataset generated by Seed Fission Technology.', metadata={'help': 'description of the dataset'})
    author: str = field(default='Zhengde Zhang', metadata={'help': 'author of the dataset'})
    version: str = field(default='1.0.0', metadata={'help': 'version of the dataset'})
    
if __name__ == '__main__':
    import hai
    args = hai.parse_args_into_dataclasses(Args)

    # args.domain = 'Neutron Science'

    domains = args.domain.split(',')
    for idx, domain in enumerate(domains):
        print(f"Process domain: {domain}")
        args.domain = domain
        main(args)

    # 备选 = ['高能物理']
    # 备选 = 'High Energy Physics,Synchrotron Radiation,Astrophysics,Neutron Science,Particle Physics'
    # 备选 = "Particle Physics,Particle Astrophysics,Experimental Physics,Theoretical Physics,Accelerator Physics,Synchrotron Radiation,Neutron Science,Computer Science"
    # domains = args.domain.split(',')
    # print('Domains: ', domains)
    # for domian in domains:
    #     run(
    #         domain=domian,
    #         seed=args.seed,
    #         num_to_gen=args.num_to_gen,
    #         max_entities=args.max_entities,
    #         output_dir=args.output_dir,
    #         model=args.model,
    #         )
    


    

