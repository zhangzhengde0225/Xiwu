
import random
import numpy as np

from xiwu.apis.xiwu_api import HepAILLM

class NewbeeBot(HepAILLM):

    def __init__(self, domain, **kwargs):
        system_prompt = f"You are a newbie in {domain} domain, learning with curiosity, asking questions based on information."
        system_prompt = f"You are a newbie, learning with curiosity, asking questions based on information."
        
        super().__init__(system_prompt, **kwargs)

    def truncate_quetions(self, max_tokens=2500):
        """将问题列表截断，使得总长度不超过max_tokens"""
        exist_questions = [qa['question'] for qa in self.data['entities']].copy()
        # reversed = exist_questions[::-1]
        def cal_qustions_token(questions):
            text = ''.join(questions)
            return self.cal_tokens(text)
        while True:
            tokens = cal_qustions_token(exist_questions)
            if tokens <= max_tokens:
                return exist_questions
            else:
                exist_questions = exist_questions[1:]

    def newbee_prompt_engineering(self, input):
        """newbee"""
        # Each question needs to provide sufficient and complete information.
        # candidate_topics = self.candidate_topics
        # if len(candidate_topics) < 2:
        #     sample_topic = candidate_topics.copy()
        # else:
        #     sample_topic = random.sample(candidate_topics, 2).copy()
        # input += '\n\n' + '\n'.join(sample_topic)
        # exist_questions = '---\n---'.join(self.truncate_quetions())

        back_prompt = 'Try to different from the existing questions delimited by triple dashes.'

        num = np.random.normal(loc=5, scale=2, size=None)
        num = round(np.clip(num, 2, 10))

        example = "{'Concepts': [<NOMINAL COMCEPT>, ...], 'Q1': '<ONE QUESTION>', 'Q2': '<ONE QUESTION>'}"
        example = '{"Q1":"<ONE QUESTION>", "Q2":"<ONE QUESTION>"}'
        # example = '{"Q1": "<ONE QUESTION>", "Q2": "<ONE QUESTION>"}'
        prompt = f"""
Please raise {num} questions from the input text delimited by triple backticks.
Prefer conceptual questioning while being specific, Each question needs to contain sufficient information and be brief.
Privoide them in JSON format, for example: 
```json
{example}
```
{self.prompt_lang}:

Input: 
```{input}```
"""
        
        
        return prompt
    
    def input2questions(self, input, **kwargs):
        """从一段文本中提出多个问题"""
        n = kwargs.get('n', 0)
        prompt = self.newbee_prompt_engineering(input)
        print('\nNewBee Prompt: \n', prompt)
        print(f'Newbee Q{n+1}: \n', end='')
        questions = self.__call__(prompt, sys_prompt=None, temperature=0.5, 
                                  response_format='json_object',  # GPT-4支持json_object
                                  **kwargs)
        return questions

    

