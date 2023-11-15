

from xiwu.apis.fastchat_api import (
    Conversation, SeparatorStyle, register_conv_template,
)


class XiwuConversation(Conversation):
    def to_hai_api_messages(self):
        ret = [{"role": "system", "content": self.system_message}]
        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret
    


xiwu_conv = XiwuConversation(
    name='xiwu',
    system_message="""
You are ChatHEP, Answer questions conversationally. Gives helpful, detailed, and polite answers to the user's questions.
""",
    roles=("USER", "ASSISTANT"),
    offset=0,
    messages=[],
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)
class VicunaConversation(Conversation):
    def to_hai_api_messages(self):
        ret = [{"role": "system", "content": self.system_message}]
        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret
    
vicuna_conv =VicunaConversation(
    name='vicuna',
    system_message="""
You are ChatHEP, Answer questions conversationally. Gives helpful, detailed, and polite answers to the user's questions.
""",
    roles=("USER", "ASSISTANT"),
    offset=0,
    messages=[],
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)

# register_conv_template(xiwu_conv, override=False)
