"""
专门用于存储chatlogs的工具。
被设计为:
- chatlog_20240101.jsonl 保存具体的chatlog文件
"""
from typing import List, Dict
import os, sys
from pathlib import Path
import json
import datetime
import atexit
import uuid
import warnings

here = Path(__file__).parent
try:
    from xiwu.version import __appname__
except:
    sys.path.insert(1, str(here.parent.parent))
    from xiwu.version import __appname__


class ChatLogs:
 
    def __init__(self) -> None:
        self.chat_logs_dir = f'{Path.home()}/.{__appname__}/chat_logs'
        self._file_path = f'{self.chat_logs_dir}/{self.get_file_name()}'
        self._data = self._init_data()
        self._register_event()

    def _register_event(self):
        import schedule
        schedule.every(10).minutes.do(self.save_to_file)
        schedule.every().day.at("00:00").do(self.swith_file_and_data)
        atexit.register(self.save_to_file)

    @property
    def data(self):
        return self._data
    
    @property
    def file_path(self):
        return self._file_path

    def _init_data(self):
        if not os.path.exists(self.chat_logs_dir):
            os.makedirs(self.chat_logs_dir, exist_ok=True)
        if not os.path.exists(self.file_path):
            data = list()
            self.save_to_file(data=data)
            return data
        # 记载数据
        with open(self.file_path, 'r') as f:
            data = []
            for line in f:
                data.append(json.loads(line))
        return data
    
    def swith_file_and_data(self, new_file_name=None):
        """每天0点时切换，启动时切换"""
        new_file_name = new_file_name if new_file_name is not None else self.get_file_name()
        new_file_path = f'{self.chat_logs_dir}/{new_file_name}'
        if new_file_path == self.file_path:
            warnings.warn("The file path is the same as the old one. do not switch.")
            return
        self.save_to_file()
        self._data = []
        self._file_path = new_file_path

    def get_file_name(self):
        """根据今日时间获取文件名"""
        current_time = datetime.datetime.now().strftime('%Y%m%d')
        return f'chatlog_{current_time}.jsonl'

    def save_to_file(self, data=None):
        data = data if data is not None else self.data
        with open(self.file_path, 'w') as f:
            for item in data:
                json_str = json.dumps(item)
                f.write(json_str + '\n')

    def append(self, entry: Dict, save_immediately=False):
        # "system_fingerprint": "fp_44709d6fcb",
        system_fingerprint = self.generate_custom_uuid(prefix="hi_", length=13)
        entry["system_fingerprint"] = system_fingerprint
        self.data.append(entry)
        if save_immediately:
            self.save_to_file()
        return system_fingerprint
        
    def generate_custom_uuid(self, prefix="hi_", length=12):
        random_uuid = uuid.uuid4()
        uuid_str = str(random_uuid).replace('-', '')
        custom_uuid = prefix + uuid_str[:length-len(prefix)]
        return custom_uuid

if __name__ == "__main__":
    cl = ChatLogs()
    one_entry = {
            "id":"chatcmpl-123",
            "object":"chat.completion.chunk",
            "created":1694268191,
            "model":"gpt-3.5-turbo-0125", 
            "choices":[
                {
                    "index":0,
                    "delta":{"role":"assistant","content":""},
                    "logprobs": None,
                    "finish_reason":None
                }
            ]
        }
    system_fingerprint = cl.append(one_entry, save_immediately=False)
    print(system_fingerprint)

    # 测试切换时间
    cl.swith_file_and_data(new_file_name='test_new_file.jsonl')
    id2 = cl.append(one_entry, save_immediately=False)
    print(id2)