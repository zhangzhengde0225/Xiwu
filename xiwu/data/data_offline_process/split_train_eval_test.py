"""
对Xiwu format数据划分训练集、验证集、测试集，划分结果存储在每个条目的`misc/used_for`字段中
"""


import os, sys
from pathlib import Path
from dataclasses import dataclass, field
here = Path(__file__).parent

import json
try:
    from xiwu.version import __version__
except:
    sys.path.append(f'{here.parent.parent.parent}')
    from xiwu.version import __version__
from xiwu.apis.xiwu_api import BaseQADatasetSaver, Entry
from xiwu.configs.constant import DATASETS_DIR



class XDatasetSplitter:

    @classmethod
    def split(cls, file_path, split_ratio=(0.8, 0.1, 0.1), shuffle=True):
        """
        split_ratio: (train_ratio, eval_ratio, test_ratio)
        """
        dataset = BaseQADatasetSaver.load_from_file(file_path)

        entries = dataset.entries
        entries_index = list(range(len(entries)))
        if shuffle:
            # 根据split_ratio划分数据集, 打断索引复制
            import random
            random.seed(429)
            random.shuffle(entries_index)

        for i, index in enumerate(entries_index):
            if i < len(entries_index) * split_ratio[0]:
                entries[index]['misc']['used_for'] = 'train'
            elif i < len(entries_index) * (split_ratio[0] + split_ratio[1]):
                entries[index]['misc']['used_for'] = 'eval'
            else:
                entries[index]['misc']['used_for'] = 'test'
        
        dataset.save()
        pass

    

if __name__ == '__main__':
    all_files = []
    datasets_dir = f'{DATASETS_DIR}/hep_text_v1.0'
    for root, dirs, files in os.walk(datasets_dir):
        for file in files:
            if file.endswith('.json'):
                all_files.append(os.path.join(root, file))
    print(f'Found {len(all_files)} json files in {datasets_dir}')
    exclude_file_names = []

    for i, file in enumerate(all_files):
        if Path(file).name in exclude_file_names:
            continue
        print(f'\rProcessing {i+1}/{len(all_files)}: {Path(file).name}', end='', flush=True)
        if Path(file).name in ["xiwu-dummy-xformat.json", 
                               "alpaca-52k-xformat.json",]:
            split_ratio = (1, 0, 0)
        else:
            split_ratio = (0.8, 0.1, 0.1)
        
        XDatasetSplitter.split(file)
    
    print('\n')
