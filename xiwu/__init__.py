
# from xiwu.apis.xiwu_api import patch_xiwu, patch_vicuna
# patch_xiwu()
# patch_vicuna()

from .version import __version__
from .apis.xiwu_api import YamlConfig

PRETRAINED_WEIGHTS_DIR = YamlConfig.PRETRAINED_WEIGHTS_DIR


