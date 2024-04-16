

from xiwu.apis.xiwu_api import HepAILLM

class ExpertBot(HepAILLM):

    def __init__(self, domain, **kwargs):
        system_prompt = f"You are an expert in {domain} domain, sharing experience, answering questions based on information, answering questions with reliable sources as much as possible"
        super().__init__(system_prompt, **kwargs)

    def question2answer(self, question, **kwargs):
        n = kwargs.get('n', 0)
        print(f'\nExpert Prompt: {question}')
        print(f'Expert A{n+1}: \n', end='')
        return self.__call__(question, **kwargs)



