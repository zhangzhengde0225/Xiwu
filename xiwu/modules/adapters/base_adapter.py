
import os, sys
from xiwu.apis.fastchat_api import (
    BaseModelAdapter
)
from xiwu import CONST


class XBaseModelAdapter(BaseModelAdapter):
    use_fast_tokenizer = True

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        # 自动修改Model_PATH
        if os.path.exists(f'{CONST.PRETRAINED_WEIGHTS_DIR}/{model_path}'):
            model_path = f'{CONST.PRETRAINED_WEIGHTS_DIR}/{model_path}'
            print(f'Using model_path: {model_path}')
        return super().load_model(model_path, from_pretrained_kwargs)

    