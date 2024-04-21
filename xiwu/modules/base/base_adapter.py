
import os, sys
from xiwu.apis.fastchat_api import (
    BaseModelAdapter
)
from xiwu import CONST


class XBaseModelAdapter(BaseModelAdapter):
    description = 'This is a Large Lanuge Model'  # 是对模型的描述
    author = "HepAI Team"  # 作者

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        # 自动修改Model_PATH
        if os.path.exists(f'{CONST.PRETRAINED_WEIGHTS_DIR}/{model_path}'):
            model_path = f'{CONST.PRETRAINED_WEIGHTS_DIR}/{model_path}'
            print(f'Using model_path: {model_path}')
        return super().load_model(model_path, from_pretrained_kwargs)

    
    