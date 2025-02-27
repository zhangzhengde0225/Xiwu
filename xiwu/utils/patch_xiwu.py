

raise DeprecationWarning('The patch_xiwu module is deprecated.')


from xiwu.apis.fastchat_api import (
    register_conv_template, register_model_adapter, conv_templates,
    model_adapters,
)
from xiwu.apis.xiwu_api import XiwuAdapter, VicunaAdapter


def patch_xiwu():
    print(f'Patching xiwu, registering xiwu_conv and XiwuAdapter...')
    # register_model_adapter(XiwuAdapter)
    model_adapters.append(XiwuAdapter)
    register_conv_template(xiwu_conv, override=True)

    
def patch_vicuna():
    print(f'Patching vicuna, registering vicuna_conv and VicunaAdapter...')
    register_model_adapter(VicunaAdapter)
    register_conv_template(vicuna_conv, override=True)
    
    
    
    