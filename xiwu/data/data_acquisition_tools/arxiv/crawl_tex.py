"""
爬取tex
"""


import requests

url = 'https://arxiv.org/e-print/2101.00011'
url = 'https://arxiv.org/e-print/2206.00865'

# 添加请求头信息
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299',
    'Referer': 'https://arxiv.org/',
    # 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept': '*/*',
    'Accept-Encoding': '*/*',
    'Connection': 'keep-alive'
}

headers = {
            # "Peferer": "https://arxiv.org/",
            "Peferer": "https://arxiv.org/e-print/2101.00011",
            # "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept": "*/*",
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
            # "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            # "Accept-Encoding": "gzip, deflate, br",
            'Accept-Encoding': '*/*',
            # "Connection": "keep-alive",
        }

# 发送请求并保存文件
response = requests.get(url, headers=headers)
with open('filename.zip', 'wb') as f:
    f.write(response.content)


