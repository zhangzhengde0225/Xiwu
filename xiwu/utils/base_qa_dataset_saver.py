
"""
基本的根据Q和A存储数据的类。
Usage:
    from base_qa_pairs_saver import BaseQADatasetSaver
    bqs = BaseQADatasetSaver()
"""
from typing import List, Union
import os, sys
from pathlib import Path
import time
import json
import damei as dm
import re
from dataclasses import dataclass, field
import schedule, atexit

here = Path(__file__).parent

import uuid


@dataclass
class Entry:
    question: str = None
    answer: str = None
    conversation: List[dict] = None
    id: str = field(default=None, metadata={"help": "id of the entry, auto generated."})
    category: str = None
    source: str = None
    labeler: str =  None
    label_time: str = None
    artificial_answer: str = None
    answer_quality: int = None
    checked: bool = False
    locked: bool = False
    misc: dict = field(default_factory=dict)


class BaseQADatasetSaver(object):
    """
    v1.1
    :param outout_file: str, default: 'formated_datasets.json'
    :kwargs: 
        version: str, default: '1.0'
        meta: dict, default: None, will be added to the meta field of the output json file.
        entities: list, each element is a data entry, default: None, will be added to the entities field of the output json file.
        other kwargs will be added to the output json file.
    """

    def __init__(
            self, 
            file_path: str = None,
            metadata: dict = None,
            save_before_exit: bool = False,
            save_every_minutes: int = 10,
            **kwargs
            ):
    
        self.file_path = file_path if file_path else f'{here}/Base-QA-data.json'
        self.data = self._load_data(metadata=metadata, **kwargs)
        self._register_auto_save(save_before_exit=save_before_exit, save_every_minutes=save_every_minutes)
        print(f'File in use: {self.file_path}')


    def _register_auto_save(self, **kwargs):
        save_before_exit = kwargs.pop('save_before_exit', True)
        save_every_minutes = kwargs.pop('save_every_minutes', 10)
        if save_before_exit:
            atexit.register(self._save_data2file, print=True)
        if save_every_minutes:
            schedule.every(save_every_minutes).minutes.do(self._save_data2file)

    def _load_data(self, **kwargs):
        file_path = self.file_path
        metadata = kwargs.pop('metadata', None)
        if metadata is None:
            metadata = {'description': 'data formated by DataFormater'}
        version = kwargs.pop('version', '1.0')
        initialize = kwargs.pop('initialize', False)
        if initialize and os.path.exists(file_path):
            ipt = input(f'File {file_path} already exists, do you want to remove and initialize it? (y/[n]): ')
            if ipt.lower() in ['y', 'yes']:
                os.remove(file_path)
        if not os.path.exists(file_path):
            file_dir = os.path.dirname(file_path)
            os.makedirs(file_dir, exist_ok=True)
            
            data_dict = dict()
            data_dict['version'] = version
            data_dict['metadata'] = metadata
            if metadata is not None:
                data_dict['metadata'].update(metadata)
            data_dict.update(kwargs)
            data_dict['entities'] = list()
            self._save_data2file(data_dict)
        else:
            with open(file_path, 'r') as f:
                data_dict = json.load(f)
            if metadata is not None:
                data_dict['metadata'].update(metadata)
            if version is not None:
                data_dict['version'] = version
        return data_dict
    
    def update_metadata(self, **kwargs):
        data = self.data
        metadata = self.metadata
        metadata['num_entities'] = len(data['entities'])
        unique_categories = list(set([x['category'] for x in data['entities'] if x['category'] is not None]))
        categories_dict = dict()
        for i, x in enumerate(data['entities']):
            c = x['category']
            if c is None:
                continue
            if c not in categories_dict:
                categories_dict[c] = 1
            else:
                categories_dict[c] += 1
        metadata['categories'] = categories_dict
        metadata['last_update'] = dm.current_time()
        self.data['metadata'] = metadata
    
    def _save_data2file(self, json_data=None, **kwargs):
        need_print = kwargs.pop('print', False)
        update_metadata = kwargs.pop('update_metadata', False)
        # check_differnce = kwargs.pop('check', False)
        data = json_data if json_data else self.data
        if need_print:
            print(f'Saving data to {self.file_path}... ', end='')
        if update_metadata:
            self.update_metadata()
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        if need_print:
            print(f', successfully!')

    @property
    def entities(self):
        return self.data['entities']
    
    @property
    def metadata(self):
        return self.data['metadata']
    
    @property
    def entries(self):
        return self.data['entities']

    def save_data2file(self, json_data=None, **kwargs):
        self._save_data2file(json_data, **kwargs)

    def save(self, json_data=None, **kwargs):
        self._save_data2file(json_data, **kwargs)

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
            completed_entity['uuid'] = entity.pop('uuid', str(uuid.uuid4().hex))

            q = entity.pop('question')
            a = entity.pop('answer')
            conv = entity.pop('conversation', None)
            if conv is not None:
                assert (q is None) and (a is None), 'q and a should be None when conversation is not None'
                completed_entity['conversation'] = conv
            else:
                completed_entity['question'] = q
                completed_entity['answer'] = a
            completed_entity['category'] = entity.pop('category', None)
            completed_entity['source'] = entity.pop('source', None)
            if 'doi' in entity:   # 额外处理一下source
                completed_entity['source'] = f'DOI: {entity.pop("doi")}'
            completed_entity['labeler'] = entity.pop('labeler', 'zdzhang')
            label_time = entity.pop('label_time', None)
            label_time = dm.current_time() if label_time is None else label_time
            completed_entity['label_time'] = label_time
            completed_entity['artificial_answer'] = entity.pop('artificial_answer', None)
            completed_entity['answer_quality'] = entity.pop('answer_quality', None)
            completed_entity['checked'] = entity.pop('checked', False)
            completed_entity['locked'] = entity.pop('locked', False)
            completed_entity['created_at'] = time.time()

            # 处理misc
            entity.pop("id")  # id已经由上面生成了， 去除掉
            if 'misc' in entity:
                completed_entity['misc'] = entity.pop('misc')
            else:
                completed_entity['misc'] = dict()
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
        # duplicate_remove = kwargs.pop('duplicate_remove', True)
        deduplication = kwargs.pop('deduplication', True)
        save_immediately = kwargs.pop('save_immediately', False)
        if deduplication:
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
        if save_immediately:
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

    def add_one_entry(self, entry: Entry, **kwargs):
        entity = entry.__dict__
        self.add_one_entity_and_save(entity, **kwargs)

    def append(self, entry: Entry, **kwargs):
        assert isinstance(entry, Entry), f'entry is not an instance of Entry: {entry}'
        self.add_one_entry(entry, **kwargs)

    def read_data(self, source):
        if source.endswith('.json'):
            with open(source, 'r') as f:
                data = json.load(f)
        elif source.endswith('.txt'):
            with open(source, 'r') as f:
                data = f.readlines()
        else:
            raise NotImplementedError(f'Not implemented for read file foramt: {source}')
        return data
        


def input_qa_one_by_one():
    file_path = f'{here.parent.parent}/data/zzd_data/qa_data.json'
    metadata = {'description': 'data inputed by zhengde zhang'}
    qadset = BaseQADatasetSaver(file_path=file_path, metadata=metadata)
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




