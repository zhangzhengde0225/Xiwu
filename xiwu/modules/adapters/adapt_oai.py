
import time
import json
# from xiwu.apis.xiwu_api import ChatLogs
from ...utils.chat_logs import ChatLogs

class OAIAdapter:
    _chat_logs: ChatLogs = None
    save_to_chat_logs = False

    @property
    def chat_logs(cls):
        if cls._chat_logs is None:
            cls._chat_logs = ChatLogs()
        return cls._chat_logs

    @classmethod
    def convert_output_to_oai_format(cls, output, model_name, stream=False):
        """
        输入格式：{
            "text": output,
            "logprobs": ret_logprobs,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": None,
        }
        输出格式：
        {
            "id":"chatcmpl-123",
            "object":"chat.completion.chunk",
            "created":1694268190,
            "model":"gpt-3.5-turbo-0125", 
            "system_fingerprint": "fp_44709d6fcb", 
            "choices":[
                {
                    "index":0,
                    "delta":{"role":"assistant","content":""},
                    "logprobs":null,
                    "finish_reason":null
                }
            ]
        }
        """
        if not stream:
            chat_complietion = OAIAdapter.build_chat_completion(output, model_name)
            if cls.save_to_chat_logs:
                uid = OAIAdapter.chat_logs.append(chat_complietion, save_immediately=False)
                chat_complietion["system_fingerprint"] = uid
            return chat_complietion
        stream_chunk = OAIAdapter.build_stream(output, model_name)
        return stream_chunk
    
    @classmethod
    def build_chat_completion(cls, output, model_name, **kwargs):
        debug = kwargs.get("debug", False)

        created = int(time.time())

        full_response = ""  # 手动添加适配OAI
        pre = 0
        for chunk in output:
            text = chunk["text"]
            usage = chunk["usage"]

            text = text.strip().split(" ")
            now = len(text) - 1
            if now > pre:
                one_token = " ".join(text[pre:now]) + " "
                if pre == 0:  # Ignore system message and qustion, 
                    pre = now
                    continue
                pre = now
                full_response += one_token
                if debug:
                    print(one_token, end='', flush=True)
            else:
                continue  # The length of text is not increasing
        last_token = " ".join(text[pre::])
        full_response += last_token
        if debug:
            print(last_token, flush=True)

        # build
        oai_chat_completion = OAIAdapter.create_chat_completion_dict(
            created, model_name, full_response, usage)
        return oai_chat_completion
            
        
    @classmethod
    def build_stream(cls, output, model_name):
        created = int(time.time())

        full_response = ""
        pre = 0
        for chunk in output:
            text = chunk["text"]
            logprobs = chunk["logprobs"]
            finish_reason = chunk["finish_reason"]
            usage = chunk["usage"]

            # Text转为Token
            text = text.strip().split(" ")
            now = len(text) - 1
            if pre == 0:  # Ignore system message and qustion, 
                pre = now
                continue
            if now <= pre:
                continue  # The length of text is not increasing
            one_token = " ".join(text[pre:now]) + " "
            pre = now
            full_response += one_token  

            # build chat completion chunk
            chunk_dict = OAIAdapter.create_chat_completion_chunk_dict(
                created, model_name, one_token, logprobs, finish_reason)
            # yield chunk_dict
            yield f'data: {json.dumps(chunk_dict)}\n\n'
        # 最后还有一个token
        last_token = " ".join(text[pre::])
        full_response += last_token
        chunk_dict = OAIAdapter.create_chat_completion_chunk_dict(
            created, model_name, last_token, logprobs, finish_reason)
        # yield chunk_dict
        yield f'data: {json.dumps(chunk_dict)}\n\n'

        # 把所有的回复合并进来，并保存
        if cls.save_to_chat_logs:
            chat_completion = OAIAdapter.create_chat_completion_dict(
                created, model_name, full_response, usage)
            uid = OAIAdapter.chat_logs.append(
                chat_completion, 
                save_immediately=True)
            # if cls.args.debug:
            #     print(f"Saved to chat logs, uid: {uid}"))


    @classmethod
    def create_chat_completion_chunk_dict(cls, created, model_name, text, logprobs, finish_reason):
        """
        创建一个chat completion chunk
        """
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "system_fingerprint": "hi_44709d6fcb",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": text
                    },
                    "logprobs": logprobs,
                    "finish_reason": finish_reason,
                }
            ]
        }

    @classmethod
    def create_chat_completion_dict(cls, created, model_name, full_response, usage):
        """
        创建一个chat completion
        """
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "system_fingerprint": "hi_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_response
                },
                "logprobs": None,
                "finish_reason": None
            }],
            "usage": {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"]
            }
        }
        

