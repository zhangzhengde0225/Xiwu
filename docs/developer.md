
# 开发者文档

## 1. 项目结构

Xiwu项目支持不同模型的统一接入，可通过命令行、API和Web三种方式使用模型。

```shell
··· run_cli.py - 运行命令行界面
··· run_worker.py - 运行模型worker，worker启动后提供相应API接入路由
··· run_webui.py - 运行WebUI，通过worker提供的API访问模型
··· ├── docs - 文档目录
··· ├── xiwu - 项目源码
··· │   ├── __init__.py
··· │   ├── apis - xiwu项目与其他项目交互的API
··· │   ├── configs - 所有配置文件，采用Python新特性dataclasses编写
··· │   ├── demo - 单个功能、模型、函数的测试代码
··· │   ├── data - 数据模块，包含数据采集、清洗、离线处理等功能
··· │   ├── modules - Xiwu项目的子模块
··· │   │   ├── __init__.py
··· │   │   ├── base - 基础模块，包括各种基础的类
··· │   │   ├── models - 模型模块，包含各种模型
··· │   │   ├── assembly_factory - 组装工厂，用于组装各种模型
··· │   │   ├── trainer - 训练器模块
··· │   │   ├── deployer - 部署器模块
··· │   ├── repos - 其他项目的代码仓库
··· │   ├── utils - 通用工具函数
··· │──────
```


## 2.提示词格式

提示词格式由`BaseConversation`来处理，不同的模型格式会不同，需要定制适配。
包含属性：name, system_message, roles, offset, sep_style, stop_str, stop_token_ids

Vicuan格式：
```bash
<SYSTEM_MESSAGE><SEP0><ROLE0>: <Q1><SEP0><ROLE1>: <A1><SEP1><>
<系统提示><一个空格>[<USER>: <问题1><一个空格><ASSISTANT>: <回答1></s>][<USER>: <问题2><一个空格><ASSISTANT>:]
例如：
"\nYou are Vicuna, Answer questions conversationally. Gives helpful, detailed, and polite answers to the user's questions.\n USER: Hello ASSISTANT: Hello there! How may I assist you today?</s>USER: who are you ASSISTANT:"
```
分隔符seps为空格`" "`和`"</s>"`,
角色roles为`USER`和`ASSISTANT`，

