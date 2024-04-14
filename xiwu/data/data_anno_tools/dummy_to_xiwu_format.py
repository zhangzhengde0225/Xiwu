


import os, sys
from pathlib import Path
import json
import ast
from dataclasses import dataclass, field
here = Path(__file__).parent

import json

try:
    from xiwu.version import __version__
except:
    sys.path.append(f'{here.parent.parent.parent}')
    from xiwu.version import __version__
from xiwu.apis.xiwu_api import BaseQADatasetSaver, Entry
from xiwu.apis.xiwu_api import HepAILLM
from xiwu.configs.constant import DATASETS_DIR
import hepai


class Dummy2Xformat(BaseQADatasetSaver):

    def __init__(self, file_path=None, **kwargs):
        version = '1.0'
        meta_data = {'description': 'The is the xiwu dummy data'}
        super().__init__(
            file_path=file_path, 
            version=version,
            meta_data=meta_data,
            **kwargs)
    
    def modify_conv(self, convs):
        """from hum"""
        raise NotImplementedError
        
    
    def __call__(self, dummy_file, **kwargs):
        data = self.read_data(dummy_file)
        for d in data:
            convs = d['conversations']
            # 修改conv的格式
            # convs  = self.modify_conv(convs)

            entry = Entry(
                conversation=convs,
                source=str(Path(dummy_file).name),
                category='dummy',
                labeler='Zhengde Zhang',
            )
            self.append(entry, deduplication=False)
        self.save(print=True)

@dataclass
class Args:
    source: str = f'{DATASETS_DIR}/raw_data/xiwu-dummy.json'
    file_path: str = f'{DATASETS_DIR}/hep_text_v1.0/xiwu-dummy-xformat.json'
    initialize: bool = True


if __name__ == '__main__':
    args = hepai.parse_args(Args)
    dummy2xformat = Dummy2Xformat(
        file_path=args.file_path,
        initialize=args.initialize
        )
    dummy2xformat(args.source)
