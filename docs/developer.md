
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

## 2.如何适配新模型到Xiwu仓库中

### 2.1. 适配逻辑

适配逻辑如图所示：
[适配逻辑图](../assets/adapter_logic.png)

关键的类如下：
+ XModel: 
    + 是支持的模型的统一入口类，例如：选择Xiwu模型，Xiwu模型会被封装成XModel，选择Vicuna模型，Vicuna模型会被封装成XModel
    + XModel被CLI、Worker使用
+ XAssembler: 
    + 是组装工厂，用于把选定的模型组装成XModel
+ Adapters: 
    + 每个模型一个适配器，用于适配模型的输入输出
    + 每个适配器包含conv：XConversation，即对话格式模板
    + 每个适配器包含

### 2.1 XModel

in xiwu/modules/base/xmodel.py
XModel是所有支持模型的统一入口，包含以下属性和方法：
+ model: 神经网络模型
+ tokenizer: 分词器
+ adapter: 适配器
+ generate_stream_func: 生成对话流的函数

上述属性由Assembler组装

### 2.2 XAssembler

in xiwu/modules/assembly_factory/assembler.py
XAssembler的核心是在初始化时注册了所有模型的Adapter，然后根据用户选择的模型，返回对应的XModel
XAseembler包含以下方法：
+ load_model: 加载模型和分词器，实际上调用Adapter的load_model方法
+ get_generate_stream_func: 获取生成对话流的函数，实际上调用Adapter的generate_stream方法或使用fastchat实现的通用函数。


### 2.3 Adapters
in xiwu/modules/adapters/*_adapter.py

每个模型的Adapter单独写成一个类，例如XiwuAdapter，集成XBaseModelAdapter
Adapter包含以下属性：
+ conv: XConversation，对话格式模板，存储了不同模型的系统提示、分隔符、角色名等，用于在构建合适的提示信息。
+ description: 模型的描述信息
+ author: 模型的作者

Adapter包含以下方法：
+ match: 根据模型的路径判断是否适合该适配器
+ generate_stream: 默认未实现，使用fastchat实现的函数；如果手动实现了，该模型会使用自定义的生成对话流函数


### 2.4 XConversation

对话格式（提示词格式）
提示词格式由`XConversation`来处理，不同的模型格式会不同，需要定制适配。
包含属性：name, system_message, roles, offset, sep_style, stop_str, stop_token_ids

通用格式(中括号`[]`是每次标识对话轮次的对话，括号本身不存在)：
```bash
<SYSTEM_MESSAGE>[<SEP0><ROLE0>: <Q1><SEP0><ROLE1>: <A1><SEP1>][<ROLE0>: <Q2><SEP0><ROLE1>:]
```
例如，在Xiwu的提示词中，分隔符seps为空格`" "`和`"</s>"`, 角色roles为`USER`和`ASSISTANT`，格式如下：
```bash
<系统提示><一个空格>[<USER>: <问题1><一个空格><ASSISTANT>: <回答1></s>][<USER>: <问题2><一个空格><ASSISTANT>:]
具体例子：
"You are Xiwu, answer questions conversationally. Gives helpful, detailed, and polite answers to the user's questions. USER: Hello ASSISTANT: Hello there! How may I assist you today?</s>USER: who are you ASSISTANT:"
```

## 3. 开发说明

适配的关键在于实现新模型的adapter，例如：llama3_adapter.py，保存到xiwu/modules/adapters/目录下，然后在Assembler中注册。

Adapter需要实现`load_model`方法实现模型的加载，`generate_stream`方法实现调用模型时对话流的生成。
Adapter需要设置`conv`属性，来适配不同模型的对话格式。
Adapter需要设置`description`和`author`属性，来描述模型的信息。

## 4. 如何测试新模型

在根目录下`python run_worker.py --model_path <新模型路径> --test`启动即可，如果已适配完成，Assembler或根据模型路径中包含的模型名匹配Adapter，然后加载模型，生成对话流。



