
"""
基本的根据Q和A存储数据的类。
Usage:
    from base_qa_pairs_saver import BaseQADatasetSaver
    bqs = BaseQADatasetSaver()
"""
import os, sys
from pathlib import Path
import json
import damei as dm
import re

here = Path(__file__).parent


class BaseQADatasetSaver(object):
    """
    :param outout_file: str, default: 'formated_datasets.json'
    :kwargs: 
        version: str, default: '1.0'
        meta: dict, default: None, will be added to the meta field of the output json file.
        entities: list, each element is a data entry, default: None, will be added to the entities field of the output json file.
        other kwargs will be added to the output json file.
    """

    def __init__(self, output_file=None, **kwargs):
        self.output_file = output_file if output_file else 'QA_datasets.json'
        self.data = self._load_data(**kwargs)
        # print(f'data: \n{dm.misc.dict2info(self.data)}')

    def _load_data(self, **kwargs):
        file_path = self.output_file
        meta_data = kwargs.pop('meta_data', None)
        if not os.path.exists(file_path):
            file_dir = os.path.dirname(file_path)
            os.makedirs(file_dir, exist_ok=True)
            
            data_dict = dict()
            data_dict['version'] = '1.0'
            data_dict['meta'] = {'description': 'data formated by DataFormater'}
            if meta_data is not None:
                data_dict['meta'].update(meta_data)
            data_dict.update(kwargs)
            data_dict['entities'] = list()
            self._save_data2file(data_dict)
            return data_dict
        else:
            with open(file_path, 'r') as f:
                data_dict = json.load(f)
            return data_dict
    
    def _save_data2file(self, json_data=None, **kwargs):
        need_print = kwargs.pop('print', False)
        data = json_data if json_data else self.data
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        if need_print:
            print(f'Save data to {self.output_file} successfully!')

    def _compeletion(self, new_entities):
        """
        补全数据
        {
            "id": 00000001,
            "question": "question text",
            "answer": "answer text",
            "category": "category name",
            "source": "HaiChatGPT",
            "labeler: "labeler name",
            "label_time": "label time",
            "artificial_answer": "artificial answer text",
            "answer_quality": 10,
            "checked": false,
            "locked": false,
            "misc": 
            {
                "questioner": "questioner name",
                "question_time": "question time",
                ...
            }
        },
        """
        completed_entities = list()
        dict2list_flag = False
        if isinstance(new_entities, dict):  # 适配单个dict的情况
            new_entities = [new_entities]
            dict2list_flag = True
        for i, entity in enumerate(new_entities):
            completed_entity = dict()
            id = len(self.data['entities']) + 1 + i
            completed_entity['id'] = f'{id:0>8}'
            completed_entity['question'] = entity.pop('question')
            completed_entity['answer'] = entity.pop('answer')
            completed_entity['category'] = entity.pop('category', None)
            completed_entity['source'] = entity.pop('source', None)
            if 'doi' in entity:   # 额外处理一下source
                completed_entity['source'] = f'DOI: {entity.pop("doi")}'
            completed_entity['labeler'] = entity.pop('labeler', 'zdzhang')
            completed_entity['label_time'] = entity.pop('label_time', dm.current_time())
            completed_entity['artificial_answer'] = entity.pop('artificial_answer', None)
            completed_entity['answer_quality'] = entity.pop('answer_quality', None)
            completed_entity['checked'] = entity.pop('checked', False)
            completed_entity['locked'] = False

            completed_entity['misc'] = dict()
            # completed_entity['misc']['questioner'] = entity.pop('questioner', None)
            # completed_entity['misc']['question_time'] = entity.pop('question_time', None)
            completed_entity['misc'].update(entity)  # 其他项目直接加入misc

            completed_entities.append(completed_entity)
        if dict2list_flag:  # 适配单个dict输入又变回去
            assert len(completed_entities) == 1, 'len(completed_entities) != 1'
            completed_entities = completed_entities[0]
        return completed_entities
    
    @property
    def exist_questions(self):
        return [x['question'] for x in self.data['entities']]

    def _duplicate_remove(self, new_entities):
        """根据问题去重"""
        clean_entities = list()
        dict2list_flag = False
        if isinstance(new_entities, dict):  # 适配单个dict的情况
            new_entities = [new_entities]
            dict2list_flag = True
        for entity in new_entities:
            if entity['question'] not in self.exist_questions:
                clean_entities.append(entity)
        if len(clean_entities) == 0:
            return None
        if dict2list_flag:
            assert len(clean_entities) == 1, f'len(clean_entities) != 1, {clean_entities}'
            clean_entities = clean_entities[0]
        
        return clean_entities
    
    def _add_entities_and_save(self, new_entities, **kwargs):
        self.data['entities'].extend(new_entities)
        self._save_data2file(**kwargs)

    def _add_one_entity_and_save(self, new_entity, **kwargs):
        duplicate_remove = kwargs.pop('duplicate_remove', True)
        if duplicate_remove:
            new_entity = self._duplicate_remove(new_entity)
        if new_entity is None:
            return
        new_entity = self._compeletion(new_entity)
        if isinstance(new_entity, dict):
            self.data['entities'].append(new_entity)
        elif isinstance(new_entity, list):
            self.data['entities'].extend(new_entity)
        else:
            raise TypeError(f'new_entity type error: {type(new_entity)}')
        self._save_data2file(**kwargs)

    def add_one_entity_and_save(self, new_entity, **kwargs):
        """
        添加一个实体并保存。
        会自动补全键名，会自动补全id，存在的键自动更新，多余的键自动加入misc.
        会自动补全标注时间
        :param: new_entity dict or list, 数据，可以是一个dict代表单条数据，也可以是一个由多个dict组成的list代表多条数据
        :param: duplicate_remove bool = True, 问题自动去重
        """
        self._add_one_entity_and_save(new_entity, **kwargs)

    def add_entities_and_save(self, entities, **kwargs):
        self._add_one_entity_and_save(entities, **kwargs)


def input_qa_one_by_one():
    file_path = f'{here.parent.parent}/data/zzd_data/qa_data.json'
    meta_data = {'description': 'data inputed by zhengde zhang'}
    qadset = BaseQADatasetSaver(output_file=file_path, meta_data=meta_data)
    count = 0
    while True:
        count += 1
        q = input(f'Q{count}: ')
        a = input(f'A{count}: ')
        if q == 'exit' or a == 'exit':
            break
        entity = dict()
        entity['question'] = q
        entity['answer'] = a
        entity['source'] = 'input from labeler'
        entity['labeler'] = 'zhengde zhang'
        qadset.add_one_entity_and_save(entity, print=True)

if __name__ == '__main__':
    input_qa_one_by_one()




