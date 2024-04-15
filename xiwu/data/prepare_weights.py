
import os, sys
from pathlib import Path
here = Path(__file__).parent

try:
    from xiwu.version import __version__
except:
    sys.path.append(f'{here.parent.parent}')
    from xiwu.version import __version__

from xiwu.configs.constant import PRETRAINED_WEIGHTS_DIR, DATASETS_DIR

