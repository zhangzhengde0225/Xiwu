import os, sys
import hepai
import ast
import json
import re
import damei as dm
from json import JSONDecodeError

logger = dm.get_logger("hepai_llm.py")

class HepAILLM:
    def __init__(self, system_prompt=None, **kwargs):
        # models = hai.Model.list()  # 列出可用模型
        # print(models)
        self._language = kwargs.get('language', 'en')
        self._system_prompt = system_prompt if system_prompt is not None else "You are ChatGPT, answering questions conversationally"
        self.model = kwargs.get('model', 'hepai/xiwu')

        self.tried = 0
        self.max_try = kwargs.get('max_try', 3)
        print(f'system_prompt: {self._system_prompt}')
        pass

    @property
    def system_prompt(self):
        return self._system_prompt
    
    @system_prompt.setter
    def system_prompt(self, value):
        self._system_prompt = value

    @property
    def prompt_lang(self):
        if self._language == 'zh':
            return 'output in Chinese'
        elif self._language == 'en':
            return 'output in English'
        else:
            raise ValueError(f'language: {self._language} is not supported')
        
    def generate(self, prompt=None, sys_prompt=None, **kwargs):
        return self.__call__(prompt=prompt, sys_prompt=sys_prompt, **kwargs)

    def __call__(self, prompt=None, sys_prompt=None, **kwargs):
        """
        请求模型， 可以提供prompt和sys_prompt
        也可以直接提供messages
        """
        try:
            return self._call(prompt=prompt, sys_prompt=sys_prompt, **kwargs)
        except Exception as e:
            print(f'Chat ERROR: {e}')
            raise e
            # return self._call(prompt=prompt, sys_prompt=sys_prompt, **kwargs)

    def _call(self, prompt=None, sys_prompt=None, **kwargs):
        # model = kwargs.pop('model', 'openai/gpt-3.5-turbo')
        model = kwargs.pop('model', self.model)
        n = kwargs.pop("n", 1)  # 移除n参数
        need_print = kwargs.get('print', True)
        system_prompt = self.system_prompt if sys_prompt is None else sys_prompt
        api_key = os.getenv('HEPAI_API_KEY', None)
        # openai_api_key = os.getenv('OPENAI_API_KEY', None)
        openai_api_key = None
        messages = kwargs.pop('messages', None)
        response_format = kwargs.pop('response_format', 'text')
        assert api_key is not None, 'api_key is None'
        assert model is not None, 'model is None'
        if messages is None:
            messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    ## 如果有多轮对话，可以继续添加，"role": "assistant", "content": "Hello there! How may I assist you today?"
                    ## 如果有多轮对话，可以继续添加，"role": "user", "content": "I want to buy a car."
                ]
        self.tried += 1
        result = hepai.LLM.chat(
                model=model,
                api_key=api_key,
                openai_api_key=openai_api_key,
                messages=messages,
                response_format=response_format,
                stream=True,
                **kwargs
            )
        # result是一个流式数据生成器，需要遍历获取全部结果
        full_result = ""
        for i in result:
            full_result += i
            if need_print:
                sys.stdout.write(i)
                sys.stdout.flush()
        if need_print:
            print()
        # 验证是否有请求错误
        if "status_code" in full_result:
            try:
                info = json.loads(full_result)
            except Exception as e:   # TODO: 可能会输出到一半，突然报错：xxx{"status_code": 42904, "message": "Too many requests"}，报错为42904报错
                pattern = r'\{.*?"status_code":\s*\d+\}'
                result = re.findall(pattern, full_result)
                assert len(result) == 1, f'len(result)={len(result)}'
                info = json.loads(result[0])
            if info['status_code'] == 42901:
                pass
            elif info['status_code'] in [42903, 42904]:
                if self.tried < self.max_try:
                    logger.info(f'请求失败：{full_result}, 重新尝试{self.tried+1}/{self.max_try}, 3秒后重试...')
                    import time
                    time.sleep(3)
                    return self._call(prompt=prompt, sys_prompt=sys_prompt, model=model, messages=messages, **kwargs)
                else:
                    raise Exception(f'请求{self.max_try}次均失败: {full_result}.')
            else:
                raise Exception(f'请求失败：{full_result}')
        self.tried = 0
        return full_result
    
    def cal_tokens(self, text, language=None):
        """大约一个汉字占2个token，一个英文单词占1.333个token"""
        lang = language if language is not None else self._language
        if lang == 'zh':
            return len(text) * 2
        elif lang == 'en':
            return len(text) * 1.333
        else:
            raise ValueError(f'language: {lang} is not supported')

    @staticmethod    
    def parse_json(topics):
        """
        解析topics,
        :param new_topics: str, "{'Q1': '<ONE QUESTION>', 'Q2': '<ONE QUESTION>'}"
        """
        try:
            new_topics = ast.literal_eval(topics)  # str2dict
        except ValueError as e:
            new_topics = topics.split('\n\n')[0]
            new_topics = ast.literal_eval(new_topics)  # str2dict
        except Exception as e:
            topics = topics.strip('```json')
            topics = topics.strip('```')
            # 去掉最开始的回车符直到出现第一个{
            topics = topics.lstrip('\n')
            topics = topics.rstrip('\n')
            try:
                new_topics = json.loads(topics)
            except JSONDecodeError as e:
                try:
                    new_topics = ast.literal_eval(topics)  # str2dict
                except Exception as e:
                    with open('error_toipc.txt', 'w') as f:
                        f.write(topics)
                    error = {'error': f'Parse topic {topics} Error: \n{e}'}
                    raise Exception(error)
        return new_topics
        
if __name__ == "__main__":
    prompt = 'hello'
    prompt = "\nPlease raise 2 questions from the input text delimited by triple backticks.\nPrefer conceptual questioning while being specific, Each question needs to contain sufficient information and be brief.\nPrivoide them in JSON format, for example: {'Q1': '<ONE QUESTION>', 'Q2': '<ONE QUESTION>'}, only output one JSON object.\noutput in Chinese\n\nInput: \n```High Energy Physics```\n"
    # ret = HepAILLM()(prompt=prompt, model='openai/gpt-3.5-turbo')
    ret = HepAILLM()(prompt=prompt, model='openai/gpt-4')

