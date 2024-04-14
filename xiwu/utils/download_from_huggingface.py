"""
从Huggingface下载llama.
默认存在当前目录下文件夹中

需要先安装git lfs并初始化
 centos安装: `sudo yum install git-lfs`
 初始化：`git lfs install`
"""

import os, sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
import hepai

here = Path(__file__).parent

try:
    from xiwu.version import __version__
except:
    sys.path.insert(1, str(here.parent.parent))
    from xiwu.version import __version__
from xiwu.configs.constant import PRETRAINED_WEIGHTS_DIR


def download(model_name, output_dir=None):
    # from transformers import AutoTokenizer,   m_pretrained(model_name)

    repo, model = model_name.split("/")
    command = f"git clone --progress https://huggingface.co/{repo}/{model}"
    output_dir = f'{output_dir}/{repo}'
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    print(command)
    print(f'The weights will be saved in {output_dir}')
    os.system(command)


@dataclass
class Args:
    model: str = field(default="lmsys/vicuna-13b-v1.5-16k", metadata={"help": "The model name to download."})
    save_dir: str = field(default=PRETRAINED_WEIGHTS_DIR, metadata={"help": "The directory to save the downloaded weights."})
    size: str = field(default=None, metadata={"help": "The size of the model."})

if __name__ == '__main__':
    args = hepai.parse_args(Args)

    # mn = "decapoda-research/llama-13b-hf"
    mn = args.model
    if args.size:
        mn = mn.replace("7b", args.size)
    # assert mn in allowed_names, f"model_name must be one of {allowed_names}"
    download(model_name=mn, output_dir=args.save_dir)



