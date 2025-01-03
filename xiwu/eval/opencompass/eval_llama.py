


import hepai
from dataclasses import dataclass, field
import copy
import getpass
import os, sys
from datetime import datetime
from pathlib import Path

from dataclasses import dataclass, field

here = Path(__file__).parent
try:
    from xiwu.version import __version__
except:
    sys.path.append(f'{here.parent.parent.parent}')
    from xiwu.version import __version__

from xiwu.eval.opencompass.interface.base_args import GeneralArgs
from xiwu.eval.opencompass.interface.eval_main import main as eval_main


@dataclass
class LLaMAArgs(GeneralArgs):
    config: str = field(default=f"{here}/configs/eval_api_llama.py", metadata={"help": "The path to the config file."})


if __name__ == "__main__":
    args = hepai.parse_args(LLaMAArgs)
    eval_main(args)
    # main(args=args) 


