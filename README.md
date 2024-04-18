
[![Stars](https://img.shields.io/github/stars/zhangzhengde0225/Xiwu)](
https://github.com/zhangzhengde0225/Xiwu)
[![Open issue](https://img.shields.io/github/issues/zhangzhengde0225/Xiwu)](
https://github.com/zhangzhengde0225/Xiwu/issues)
[![Datasets](https://img.shields.io/static/v1?label=Download&message=source_code&color=orange)](
https://github.com/zhangzhengde0225/Xiwu/archive/refs/heads/main.zip)

#### English | [简体中文](https://github.com/zhangzhengde0225/Xiwu/blob/main/docs/README_zh_cn.md)

# HEP·Xiwu LLM

<div align="center">
  <p>
    <a href="https://ai.ihep.ac.cn/m/xiwu" target="_blank">
      <img width="70%" src="/assets/GA.png" alt="Graphical Abstract"></a>
  </p>
</div>

This is the first LLM for HEP, an offitial implemention of [Xiwu(溪悟): A Basis Flexible and Learnable LLM for High Energy Physics](https://arxiv.org/abs/2404.08001). This model is designed to possess exceptional capabilities in common sense answering, BOSS code generation, and physical logical reasoning.

Xi(溪): stremlet → drops of water, Wu(悟): understand and gaining insight

# Features

+ Xiwu, the first LLM specilized for high energy physics outperforms the foundation model in accuracy for domain-specific knowledge question answering, and exceeds GPT-4 in BOSS (BESIII Offline Software System) code
+ Xiwu is a Level 2 model that can smoothly switch between foundation models such as LLaMA, Vicuna, ChatGLM and Grok-1.
+ Xiwu equipped with two learning systems: The Just-In-Time Learning system based on RAG is capable of acquiring new knowledge instantly, and the On-The-Fly Traning system based on secondary pre-training and fine-tuning can be used to enhance the model's performance in specific tasks.

# Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```
You can see the basic configurations in the [configs.py](xiwu/configs/configs.py) and [constant.py](xiwu/configs/constant.py) files.

## Deploy
+ CLI Demo  
```bash 
python run_xiwu_cli.py \
  --model_path xiwu/xiwu-13b-20230509 \
  --load_8bit False 
```
For more available arguments, you can run `python run_xiwu_cli.py -h`.

+ Deploy worker to a server
You should run a controller befor you run the worker.
```bash
python run_xiwu_worker.py
```

### Trained Weights

Refer to the [models](docs/models.md) for the trained weights.

## Train on Custom Data

```bash
bash scripts/train_xiwu.sh 
```

## Performance Comparison

<div align="center">
  <p>
    <img width="80%" src="/assets/hallucination.png" alt="Comparision"></a>
  </p>
  <p>Comparsion of GPT-4 and Xiwu in HEP Kownledge Q&A and BOSS Code Generation</p>
</div>


# Contributors

Xiwu is authored by Zhengde Zhang, Yiyu Zhang, Haodong Yao, Jianwen Luo, Rui Zhao, Bo Huang, Jiameng Zhao, Yipu Liao, Ke Li, Lina Zhao, Fazhi Qi and Changzheng Yuan. 

Currently, it is maintained by Zhengde Zhang (zdzhang@ihep.ac.cn).

# Acknowledgements

This work is Supported by the Informatization Plan of Chinese Academy of Science, Grant
No. CAS-WX2022SF-0104 and "From 0 to 1" Original Innovation Project of IHEP, Grant No. E3545PU2.
We would like to express our gratitude to Beijiang Liu, Yaquan Fang, Gang Li, Wuming Luo, Ye Yuan, Shengsen Sun, Yi Jiao and others who are not listed here for engaging in beneficial discussions or providing computing resources.

We are very grateful to the [LLaMA](https://github.com/meta-llama/llama), [FastChat](https://github.com/lm-sys/FastChat) projects for the foundation models.

# Citation
```
@misc{zhang2024xiwu,
      title={Xiwu: A Basis Flexible and Learnable LLM for High Energy Physics}, 
      author={Zhengde Zhang and Yiyu Zhang and Haodong Yao and Jianwen Luo and Rui Zhao and Bo Huang and Jiameng Zhao and Yipu Liao and Ke Li and Lina Zhao and Fazhi Qi and Changzheng Yuan},
      year={2024},
      eprint={2404.08001},
      archivePrefix={arXiv},
      primaryClass={hep-ph}
}
```

# License

This project is licensed under the terms of the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.