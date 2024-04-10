import os, sys
from pathlib import Path
here = Path(__file__).parent

## 适配hepai后端
try:
    import hepai
except:
    hepai_dir = f'{here.parent.parent.parent.parent}/hai'
    print(f'Not found `hepai`, use the library in `{hepai_dir}`.')
    sys.path.insert(1, str(hepai_dir))
    import hepai


