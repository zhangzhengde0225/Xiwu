
import os, sys
from pathlib import Path
here = Path(__file__).parent

from dataclasses import dataclass, field
try:
    from xiwu.version import __version__
except:
    sys.path.append(f'{here.parent.parent}')
    from xiwu.version import __version__

from xiwu.configs.constant import PRETRAINED_WEIGHTS_DIR, DATASETS_DIR
import hepai


class PrepareWeights:
    supported_models = [
        "lmsys/vicuna-7b-v1.5",
        "lmsys/vicuna-7b-v1.5-16k",
        "lmsys/longchat-7b-v1.5-32k",
        "lmsys/vicuna-13b-v1.5",
        "lmsys/vicuna-13b-v1.5-16k",
        "lmsys/vicuna-33b-v1.3",
        "hepai/xiwu-13b-16k-20240417",
        ]

    def __init__(self, args, **kwargs):
        self.args = args

    def __call__(self, **kwargs):
        if self.args.list_all:
            print("Supported models:")
            for model in self.supported_models:
                print(f"    {model}")
            return
        model = self.args.model
        save_dir = self.args.save_dir
        if os.path.exists(f'{save_dir}/{model}'):
            print(f'{model} already exists in {save_dir}')
            return
        ok = self.download_from_huggingface(model, save_dir)
        if not ok:
            print(f"Download {model} failed.")
            return

    def download_from_huggingface(self, model_name, output_dir=None):
        repo, model = model_name.split("/")
        command = f"git clone --progress https://huggingface.co/{repo}/{model}"
        output_dir = f'{output_dir}/{repo}'
        try:
            os.makedirs(output_dir, exist_ok=True)  # 创建目录
            os.chdir(output_dir)  # 切换目录
            print(f'Excute command `{command}`')  # 打印命令
            print(f'The weights will be saved in `{output_dir}`')
            os.system(command)
        except Exception as e:
            print(e)
            return False
        return True


@dataclass
class Args:
    model: str = field(default="lmsys/vicuna-7b-v1.5-16k", metadata={"help": "The model name to download."})
    save_dir: str = field(default=PRETRAINED_WEIGHTS_DIR, metadata={"help": "The directory to save the downloaded weights."})
    list_all: bool = field(default=False, metadata={"help": "List all supported models."})

    def __post_init__(self):
        self.save_dir = Path(self.save_dir).resolve()

if __name__ == '__main__':
    args = hepai.parse_args(Args)

    pw = PrepareWeights(args)
    pw()
