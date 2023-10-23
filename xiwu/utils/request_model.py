import os, sys
import hai
hai.api_key = os.getenv('HEPAI_API_KEY')

models = hai.Model.list()  # 列出所有可用模型
print(models)

def request_model(prompt='hello', system_prompt=None):
    # system_prompt = system_prompt if system_prompt else "Answering questions conversationally"
    # system_prompt = None

    result = hai.LLM.chat(
            # model='openai/gpt-3.5-turbo',
            # model='hepai/xiwu-13b-0509',
            # model='lmsys/vicuna-13b',
            model='hepai/vicuna-7B',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                ## 如果有多轮对话，可以继续添加，"role": "assistant", "content": "Hello there! How may I assist you today?"
                ## 如果有多轮对话，可以继续添加，"role": "user", "content": "I want to buy a car."
            ],
            stream=True,
        )

    full_result = ""
    for i in result:
        full_result += i
        sys.stdout.write(i)
        sys.stdout.flush()
        # import time
        # time.sleep(0.5)
    print()
    return full_result

prompt = 'who are you?'
# prompt = "what's your name?"
answer = request_model(prompt=prompt)
print(answer)