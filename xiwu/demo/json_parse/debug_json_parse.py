

from pathlib import Path
here = Path(__file__).parent

file = f'{here.parent}/error_toipc1.txt'

with open(file, 'r') as f:
    topics = f.read()

import ast
import numpy as np
import json


print(topics, type(topics))
a = [topics]

print(a)

## 正则表达式匹配{"xxx"status_code": xxx}的内容
import re
pattern = r'\{.*?"status_code":\s*\d+\}'
result = re.findall(pattern, topics)
assert len(result) == 1, f'len(result)={len(result)}'  # 有且只有一个匹配
topics = result[0]

data = json.loads(topics)

# data = ast.literal_eval(a[0])

# b = {'Q1':'How does Kepler's second law illustrate the principle of conservation of angular momentum in planetary motion?', 'Q2':'Why must the velocity of a planet increase as it gets closer to the Sun according to the conservation of angular momentum?'}


print(data, type(data))
print(data.keys())

