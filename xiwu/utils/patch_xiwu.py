
from xiwu.apis.fastchat_api import register_conv_template, register_model_adapter, conv_templates
from xiwu.apis.xiwu_api import xiwu_conv, XiwuAdapter,vicuna_conv,VicunaAdapter

def patch_xiwu():
    print(f'Patching xiwu, registering xiwu_conv and XiwuAdapter...')
    register_model_adapter(XiwuAdapter)
    register_conv_template(xiwu_conv, override=True)
def patch_vicuna():
    print(f'Patching vicuna, registering vicuna_conv and VicunaAdapter...')
    register_model_adapter(VicunaAdapter)
    register_conv_template(vicuna_conv, override=True)
    
    
    
    