from hepai import HepAI


client = HepAI(base_url="http://localhost:8000/v1")

a = client.models.list()
print(a)

rst = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "system",
            "content": "You are a chatbot named Xiwu."
        },
        {
            "role": "user",
            # "content": "sai hello"
            # "content": "你是谁"
            "content": "tell me a story"
        }
    ]
)

print(rst)