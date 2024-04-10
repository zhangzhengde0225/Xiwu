




import requests
import json


def request_completion(stream=False):
    url = "http://localhost:42902/completion"
    url = "https://aiapi.ihep.ac.cn/inference"
    messages = [
        {"role": "system", "content": "You are a elpful AI Assistant."},
        {"role": "user", "content": "what is meson?"},
    ]
    data = {
        # "model": "gpt-3.5-turbo",
        # "model": "gpt-4-turbo-preview", # "moonshot-v1-32k
        "model": "lmsys/vicuna-7b",
        "messages": messages,
        "temperature": 0.5,
        "stream": stream,
    }
    response = requests.post(
        url, 
        json=data,
        stream=stream,
        )
    
    if response.status_code != 200:
        print(response.status_code, response.text)
        raise Exception(f"Failed to connect to the server.")
    
    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode('utf-8').strip()
        js_line = line[6::] if line.startswith("data: ") else line
        chunk = json.loads(js_line)
        if chunk.get("choices"):
            finish_reason = chunk["choices"][0]['finish_reason']
            delta = chunk["choices"][0]['delta']
            content = delta.get("content")
            role = delta.get("role")
            function_call = delta.get("function_call")
            tool_calls = delta.get("tool_calls")
            if finish_reason == "stop":
                break
            # print(f'{content}', end='', flush=True)
            yield content


if __name__ == "__main__":
    res = request_completion()
    for r in res:
        print(r, end="", flush=True)
    print()











