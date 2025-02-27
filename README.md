[![Stars](https://img.shields.io/github/stars/zhangzhengde0225/Xiwu)](
https://github.com/zhangzhengde0225/Xiwu)
[![Open issue](https://img.shields.io/github/issues/zhangzhengde0225/Xiwu)](
https://github.com/zhangzhengde0225/Xiwu/issues)
[![Datasets](https://img.shields.io/static/v1?label=Download&message=source_code&color=orange)](
https://github.com/zhangzhengde0225/Xiwu/archive/refs/heads/main.zip)

#### English | [简体中文](https://github.com/zhangzhengde0225/Xiwu/blob/main/docs/README_zh_cn.md)

<h1><img src="/assets/xiwu.png" alt="xiwu logo" style="width: 30px"> HEP·Xiwu LLM </h1>

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

### Prepare trained Weights

By default, the model weights will be stored in the `/data/<USERNAME>/weights` directory, you set the `PRETRAINED_WEIGHTS_DIR` cont in the [constant.py](xiwu/configs/constant.py) file or the `PRETRAINED_WEIGHTS_DIR` environment variable to change the default directory.

You can run `./prepare_weights.sh --list_all` to see all available weights, and run the following command to download the trained weights:
```bash
./prepare_weights.sh --model lmsys/vicuna-7b-v1.5 
```

## Deploy
#### Run CLI (Command Line Interface) to interact with the model 
```bash 
python run_cli.py \
  --model_path xiwu/xiwu-13b-16k-20240417 \
  --load_8bit False 
```
You and switch to any supported model. For more available arguments, you can run `python run_cli.py -h`.
The assembler will automatically search the model in the `PRETRAINED_WEIGHTS_DIR` directory.

### Deploy a worker to host an API server
```bash
python run_worker.py \
  --model_path xiwu/xiwu-13b-16k-20240417 \
```
For more available arguments, you can run `python run_worker.py -h`.

After the worker is started, you can open a new terminal and access the model via by following script:
```bash
python request_api.py
```
Note that you should specify the `base_url` in the script to the address of the worker.
Streaming API is also supported in this script.


## Train on Custom Data to get a new model

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

If you are interested in contributing to Xiwu, please refer to the [Contributing Guidelines](docs/developer.md).

Currently, `Xiwu` is authored by Zhengde Zhang, Yiyu Zhang, Haodong Yao, Jianwen Luo, Rui Zhao, Bo Huang, Jiameng Zhao, Yipu Liao, Ke Li, Lina Zhao, Fazhi Qi and Changzheng Yuan. 

it is maintained by Zhengde Zhang (zdzhang@ihep.ac.cn).

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
