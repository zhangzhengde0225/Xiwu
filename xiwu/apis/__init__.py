import os, sys
from pathlib import Path
here = Path(__file__).parent

sys.path.append(f'{here.parent}/repos/FastChat')

from .fastchat_api import *