


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

class HaiChat19667_TO_Xformat(BaseQADatasetSaver):

    def __init__(self, file_path=None, **kwargs):
        version = '1.1'
        metadata = {
            'description': 'Real concerns of the HEP researchers from HaiChat',
            'source': 'HaiChat (http://chat.ihep.ac.cn)',
        }
        super().__init__(
            file_path=file_path, 
            version=version,
            metadata=metadata,
            **kwargs)
    
    
    def __call__(self, source_file, **kwargs):
        data = self.read_data(source_file)
        for i, d in enumerate(data['entities']):
            # 修改conv的格式
            # convs  = self.modify_conv(convs)
            id = d.pop("id")
            print(f'\rprocessing {i+1:0>5}/{len(data["entities"])}', end='', flush=True)
            entry = Entry(
                question=d.pop('question'),
                answer=d.pop('answer'),
                answer_quality=d.pop('answer_quality', None),
                category=d.pop('category', None),
                labeler=d.pop('labeler', None),
                label_time=d.pop('label_time', None),
                locked=d.pop('locked'),
                artificial_answer=d.pop('artificial_answer', None),
                checked=d.pop('checked', False),
            )
            misc = dict()
            misc.update(d)
            entry.misc = misc

            self.append(entry, deduplication=False)
        print()
        # 统计一下各类有多少个
        self.save(print=True, update_metadata=True)

@dataclass
class Args:
    source: str = f'{DATASETS_DIR}/raw_data/HaiChat-qa-19667.json'
    file_path: str = f'{DATASETS_DIR}/hep_text_v1.0/haichat-qa-19667.json'
    initialize: bool = True


if __name__ == '__main__':
    args = hepai.parse_args(Args)
    h2x = HaiChat19667_TO_Xformat(file_path=args.file_path, initialize=args.initialize)
    h2x(args.source)
