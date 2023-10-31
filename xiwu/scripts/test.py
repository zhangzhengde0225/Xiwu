import hai
import os , sys
#hai.api_key = os.getenv("HEPAI_API_KEY")
hai.api_key = "Hi-kqrKJvCdMgGiAoTBabZHVISSRMiOlVnxosGFfbhOVDUcveY"
models = hai.Model.list()  # 列出所有可用模型
print(models)
prompt='hello, what is your name?'
system_prompt=None
system_prompt = system_prompt if system_prompt else "Answering questions conversationally"
def request_model(prompt, system_prompt=None):
    system_prompt = system_prompt if system_prompt else "Answering questions conversationally"
    result = hai.LLM.chat(
        #model='hepai/xiwu-13B',
        #model='hepai/vicuna-7B',
        #model='hepai/vicuna-13B',
        #model='hepai/vicuna-33B',
        #model='hepai/vicuna-7B-v1.5',
        #model='hepai/llama-7B',
        model='hepai/llama-7B-v2',
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
    print()
    return full_result

answer = request_model(prompt=prompt)