
from dataclasses import dataclass
import hepai

@dataclass
class BaseArgs:
    model_path: str = f'chathep/chathep-13b-20230509' 
    device: str = "cuda"

@dataclass
class ModelArgs(BaseArgs):
    model_path: str = f'chathep/chathep-13b-20230509-2'


args1 = hepai.parse_args(BaseArgs)
args2 = hepai.parse_args(ModelArgs)

print(args1)
print(args2)
