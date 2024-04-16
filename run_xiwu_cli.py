from typing import Optional, List
import os, sys
from pathlib import Path
here = Path(__file__).parent

from dataclasses import dataclass, field
try:
    from xiwu.version import __version__
except:
    sys.path.append(f'{here.parent.parent.parent}')
    from xiwu.version import __version__
from xiwu.apis.fastchat_api import main
from xiwu.configs import BaseArgs
from xiwu.modules.serve.cli import CLI
import hepai

@dataclass
class ModelArgs(BaseArgs):
    model_path: str = field(default="xiwu/xiwu-13b-20230509", metadata={"help": "The path to the weights. This can be a local folder or a Hugging Face repo ID."})
    debug: bool = False
    
if __name__ == "__main__":
    args = hepai.parse_args(ModelArgs)
    print(args)
    # main(args)
    
    CLI.main(args)
    