#coding=utf-8

"""
提取BOSS代码
"""


import os, sys
from pathlib import Path
import shutil
import json

here = Path(__file__).parent

class BossCodeExtract:
    
    def __init__(self) -> None:
        self.save_file_path = f"{here}/Boss_7.1.0_codes/Boss_7.1.0_metadata.json"
        self._data = self._init_data()
        self.sufixes = [".h", ".hh", ".cc", ".c", ".cxx", ".cpp", ".C", ".txt", ".xml", ".sh", ".dat", ".dec", ".f"]
        # TODO: 包含requirements, 没有后缀的文件
        # 大写的文件名, 如.DEC
        # .table
        pass
    
    @property
    def data(self):
        if self._data is None:
            self._data = self._init_data()
        return self._data
    
    
    def _init_data(self):
        if os.path.exists(self.save_file_path):
            return self._load_data()
        else:
            data = dict()
            data['version'] = "7.1.0"
            data['metadata'] = dict()
            data['metadata']['description'] = "BOSS 7.1.0 codes"
            data["metadata"]["code_save_dir"] = f'{Path(self.save_file_path).parent}/src'
            data['entities'] = list()
            self._save_data(data)
            return data
            
    def _save_data(self, data=None):
        data = data or self.data
        with open(self.save_file_path, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
    def _load_data(self):
        with open(self.save_file_path, 'r') as f:
            data = json.load(f)
        return data
    
    
    def __call__(self, search_root):
        
        
        count = 0
        for root, dirs, files in os.walk(search_root):
            for file in files:
                if file.endswith(tuple(self.sufixes)):
                    file_path = os.path.join(root, file)
                    
                    save_dir = f'{Path(self.save_file_path).parent}/src'
                    save_file_path = file_path.replace(search_root, save_dir)
                    self.save_file(file_path, save_file_path)
                    # 保存到metadata
                    self.save_metadata(save_file_path, save_dir)
                    count += 1
                    print(f"\r{count}", end='')
        print()
        self._save_data()
        pass
    
    def save_metadata(self, save_file_path, save_dir):
        d = self.data
        
        if save_file_path not in d['entities']:
            # relative_path = Path(save_file_path).relative_to(save_dir)
            relative_path = save_file_path.replace(save_dir, "")
            d['entities'].append(str(relative_path))
            # self._save_data()
    
    def save_file(self, file_path, save_file_path):
        
        save_dir = os.path.dirname(save_file_path)
        os.makedirs(save_dir, exist_ok=True)
        
        if os.path.exists(save_file_path):
            return
        shutil.copyfile(file_path, save_file_path)
        # print(file_path, save_file_path)
        
    def analyse(self):
        """分析想要的数据"""
        metadata = self.data['metadata']
        entities = self.data['entities']
        # 各种类别有多少
        num_each_type = dict()
        num_of_words = 0
        unreadable_files = list()
        for i, entitie in enumerate(entities):
            # suffix = os.path.splitext(entities)[1]
            suffix = str(Path(entitie).suffix)
            if suffix == "":  # 适配xxx/.txt没有名字的情况
                suffix = f'.' + entitie.split(".")[-1]
                # print(entities)
            if suffix not in num_each_type:
                num_each_type[suffix] = 1
            else:
                num_each_type[suffix] += 1
            file_path = f'{metadata["code_save_dir"]}{entitie}'
            with open(file_path, 'r') as f:
                try:
                    content = f.read()
                except:
                    if entitie not in unreadable_files:
                        unreadable_files.append(entitie)
            words = content.split()
            num_of_words += len(words)
            print(f"\r{i}", end='')
        print()
            
        metadata["num_of_words"] = num_of_words
        metadata["num_of_files"] = len(entities)
        metadata["num_each_type"] = num_each_type
        metadata["unreadable_files_desc"] = "Empty or Contain grabled codes"
        metadata["unreadable_files"] = unreadable_files
        self._save_data()
    

if __name__ == "__main__":
    search_root = "/cvmfs/bes3.ihep.ac.cn/bes3sw/Boss/7.1.0"
    bce = BossCodeExtract()
    # bce(search_root=search_root)
    bce.analyse()