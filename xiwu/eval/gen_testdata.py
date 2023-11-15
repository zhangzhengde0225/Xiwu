import string
import json
import random
import os
with open("/home/linshan/xiwu/data/xiwu_eval_dataset/test_data.txt", 'r') as f:
    content = f.readlines()       
my_dict = {}
ref_answer = {}
i=1
Q_id=0
list=[]
while  i <= (len(content)-2):
    key_Qvalue = content[i].strip().split(': ')
    key_Avalue= content[i+1].strip().split(': ')
    i=i+3
    if 'Q' in key_Qvalue:
        if key_Qvalue not in list: #去除重複question
            list.append(key_Qvalue)
            Q_id=Q_id+1
            my_dict['question_id']=Q_id
            my_dict["category"]='hep'
            my_dict['turns'] = [key_Qvalue[1]]
            with open('test_question_data.jsonl', 'a+') as f:
                json.dump(my_dict, f)
                f.write('\n')
                
            ref_answer["question_id"] = Q_id
            random_string = ''.join(random.choice(string.ascii_letters) for _ in range(22))
            ref_answer["answer_id"] = random_string
            ref_answer["model_id"] = "gpt-4"
            ref_answer["choices"] =[{"index": 0, "turns": [key_Avalue[1]]}]
            integer_part = random.randint(1e9, 1e10 - 1)  # 1e9表示10^9
            # 生成6位小数部分
            decimal_part = random.randint(1, 999999) / 1e6
            # 合并整数和小数部分
            random_number = integer_part + decimal_part
            ref_answer["tstamp"] = random_number
            with open('test_Ref_Answer_data.jsonl', 'a+') as f:
                json.dump(ref_answer, f)
                f.write('\n')