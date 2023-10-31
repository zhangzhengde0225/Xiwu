import os, sys
import hai


class HepAILLM:
    def __init__(self, system_prompt=None, **kwargs):
        # models = hai.Model.list()  # 列出可用模型
        # print(models)
        self._language = kwargs.get('language', 'en')
        self.system_prompt = system_prompt if system_prompt is not None else "You are ChatGPT, answering questions conversationally"
        print(f'system_prompt: {self.system_prompt}')
        pass

    @property
    def prompt_lang(self):
        if self._language == 'zh':
            return 'output in Chinese'
        elif self._language == 'en':
            return 'output in English'
        else:
            raise ValueError(f'language: {self._language} is not supported')

    def __call__(self, prompt, sys_prompt=None, **kwargs):
        try:
            return self._call(prompt, sys_prompt=sys_prompt, **kwargs)
        except Exception as e:
            print(f'Chat ERROR: {e}')
            return self._call(prompt, sys_prompt=sys_prompt, **kwargs)

    def _call(self, prompt, sys_prompt=None, **kwargs):
        # model = kwargs.pop('model', 'openai/gpt-3.5-turbo')
        model = kwargs.pop('model')
        kwargs.pop('n', None)
        need_print = kwargs.get('print', True)
        # prompt = "Hello!"
        system_prompt = self.system_prompt if sys_prompt is None else sys_prompt
        api_key = os.getenv('HEPAI_API_KEY', None)

        result = hai.LLM.chat(
                model=model,
                api_key=api_key,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    ## 如果有多轮对话，可以继续添加，"role": "assistant", "content": "Hello there! How may I assist you today?"
                    ## 如果有多轮对话，可以继续添加，"role": "user", "content": "I want to buy a car."
                ],
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
        
if __name__ == "__main__":
    prompt = 'hello'
    prompt = "\nPlease raise 2 questions from the input text delimited by triple backticks.\nPrefer conceptual questioning while being specific, Each question needs to contain sufficient information and be brief.\nPrivoide them in JSON format, for example: {'Q1': '<ONE QUESTION>', 'Q2': '<ONE QUESTION>'}, only output one JSON object.\noutput in Chinese\n\nInput: \n```High Energy Physics```\n"
    HepAILLM()(prompt=prompt, model='openai/gpt-3.5-turbo')