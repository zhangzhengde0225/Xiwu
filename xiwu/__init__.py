
# from xiwu.apis.xiwu_api import patch_xiwu, patch_vicuna
# patch_xiwu()
# patch_vicuna()

from .version import __version__
from .configs import constant as CONST
from .configs.basic_config import BaseArgs
from .modules.assembly_factory.assembler import XAssembler
ASSEMBLER = XAssembler()

from .modules.models.base_model import XBaseModel, XBaseModelArgs
from .apis.xiwu_api import YamlConfig

PRETRAINED_WEIGHTS_DIR = YamlConfig.PRETRAINED_WEIGHTS_DIR

