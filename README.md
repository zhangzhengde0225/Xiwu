
[![Stars](https://img.shields.io/github/stars/zhangzhengde0225/Xiwu)](
https://github.com/zhangzhengde0225/Xiwu)
[![Open issue](https://img.shields.io/github/issues/zhangzhengde0225/Xiwu)](
https://github.com/zhangzhengde0225/Xiwu/issues)
[![Datasets](https://img.shields.io/static/v1?label=Download&message=source_code&color=orange)](
https://github.com/zhangzhengde0225/Xiwu/archive/refs/heads/main.zip)

#### English | [简体中文](https://github.com/zhangzhengde0225/Xiwu/blob/main/docs/README_zh_cn.md)

# Xiwu

<div style="text-align: center;"><img src="/assets/GA.png" width="70%" height="50%" /></div>

The **HEP·Xiwu(溪悟)** a basis flexible and learnable Large Language Model (LLM) tailored for High Energy Physics (HEP) research field. This model is designed to possess exceptional capabilities in **common sense answering**, **BOSS code generation**, and **physical logical reasoning**.

Xi(溪): stremlet → drops of water, Wu(悟): understand and gaining insight

# Features

+ Xiwu, the first LLM specilized for high energy physics outperforms the foundation model in accuracy for domain-specific knowledge question answering, and exceeds GPT-4 in BOSS (BESIII Offline Software System) code
+ Xiwu is a Level 2 model that can smoothly switch between foundation models such as LLaMA, Vicuna, ChatGLM and Grok-1.
+ Xiwu equipped with two learning systems: **Just-In-Time Learning** system based on RAG is capable of acquiring new knowledge instantly, and **On-The-Fly Traning** system based on secondary pre-training and fine-tuning can be used to enhance the model's performance in specific tasks.

# Quick Start

## Train

```bash
bash scripts/train_xiwu.sh 
```

## Deploy
+ CLI Demo  
```bash 
bash scripts/run_xiwu_cli.sh
```
+ Deploy worker
```bash
bash scripts/deploy_xiwu_worker.sh
```
The worker serves as the backend of the webui, and you can access the worker through the API.
+ Deploy webui
```bash
bash scripts/deploy_webui.sh
```
The webui is the front end of the system, and you can access the webui through the browser.

## Trained Weights

Refer to the [models](docs/models.md) for the trained weights.

# Contributors

Xiwu is authored by Zhengde Zhang, Yiyu Zhang, Haodong Yao, Jianwen Luo, Rui Zhao, Bo Huang, Jiameng Zhao, Yipu Liao, Ke Li, Lina Zhao, Jun Cao, Fazhi Qi and Changzheng Yuan. 

Currently, it is maintained by Zhengde Zhang (zdzhang@ihep.ac.cn).

# Acknowledgements

This work is Supported by the Informatization Plan of Chinese Academy of Science, Grant
No. CAS-WX2022SF-0104 and "From 0 to 1" Original Innovation Project of IHEP, Grant No. E3545PU2.
We would like to express our gratitude to Beijiang Liu, Yaquan Fang, Gang Li, Wuming Luo, Ye Yuan, Shengsen Sun, Yi Jiao and others who are not listed here for engaging in beneficial discussions or providing computing resources.

# License

This project is licensed under the terms of the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.