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
from xiwu import BaseArgs
from xiwu.modules.serve.cli import CLI
import hepai

@dataclass
class ModelArgs(BaseArgs):
    model_path: str = field(default="lmsys/vicuna-7b-v1.5-16k", metadata={"help": "The path to the weights. This can be a local folder or a Hugging Face repo ID."})
    debug: bool = True
    
if __name__ == "__main__":
    args = hepai.parse_args(ModelArgs)
    print(args)
    CLI.main(args)
    