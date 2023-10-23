
from xiwu.apis.fastchat_api import register_conv_template, register_model_adapter, conv_templates
from xiwu.apis.xiwu_api import xiwu_conv, XiwuAdapter

def patch_xiwu():
    print(f'Patching xiwu, registering xiwu_conv and XiwuAdapter...')
    register_model_adapter(XiwuAdapter)
    register_conv_template(xiwu_conv, override=True)
    
    
    