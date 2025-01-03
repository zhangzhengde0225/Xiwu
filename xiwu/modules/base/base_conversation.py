


try:
    from xiwu.apis.fastchat_api import Conversation
except:
    Conversation = object
    pass


class XConversation(Conversation):
    """
    This is the base conversation class for all models.
    """

    def to_hepai_api_messages(self):
        ret = [{"role": "system", "content": self.system_message}]
        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret






