

##  第一章：AI 大模型四阶技术总览

-   深度解读 AI 发展四轮浪潮

-   技术浪潮：弱人工智能、机器学习、深度学习、大语言模型

-   应用浪潮：高校共识、硅谷创新、中美博弈

-   把握浪潮：AI 大模型助力超级个体和小团队

-   AI 大模型四阶技术总览

-   提示工程（Prompt Engineering）

-   AI 智能体（Agents）

-   大模型微调（Fine-tuning）

-   预训练技术（Pre-training）

##  第二章：大语言模型技术发展与演进

-   统计语言模型

-   神经网络语言模型

-   基于 Transformer 的大语言模型

-   注意力机制

-   Transformer 网络架构

-   预训练 Transformer 模型： GPT-1 与 BERT

-   暴力美学 GPT 系列模型

##  第三章：大模型开发工具库 Hugging Face Transformers

-   Hugging Face Transformers 快速入门

-   Transformers 库是什么？

-   Transformers 核心功能模块

-   使用 Pipelines 快速实践大模型

-   使用 Tokenizer 编解码文本

-   使用 Models 加载和保存模型

-   大模型开发环境搭建

-   搭建你的 GPU 开发环境

-   Google Colab 测试环境

-   实战 Hugging Face Transformers 工具库

##  第四章：实战 Transformers 模型训练

-   数据集处理库 Hugging Face Datasets

-   Hugging Face Datasets 库简介

-   数据预处理策略：填充与截断

-   使用 Datasets.map 方法处理数据集

-   Transformers 模型训练入门

-   模型训练基类 Trainer

-   训练参数与超参数配置 TrainingArguments

-   模型训练评估库 Hugging Face Evaluate

-   实战使用 Transformers 训练 BERT 模型

-   bert-base-cased 模型（文本分类任务）

-   distilbert-base-uncased 模型（QA 任务）

##  第五章：大模型高效微调技术揭秘（上）

-   Before PEFT：Hard Prompt / Full Fine-tune

-   PEFT 主流技术分类介绍

-   PEFT Adapter 技术

-   Adapter Tuning （2019 Google）

-   PEFT – Soft Prompt 技术（Task-specific Tuning）

-   Prefix Tuning （2021 Stanford）

-   Prompt Tuning （2021 Google）

-   PEFT – Soft Prompt 技术（Prompt Encoder）

-   P-Tuning v1 （2021 Tsinghua, MIT）

-   P-Tuning v2 （2022 Tsinghua, BAAI, Shanghai Qi Zhi Institute）

##  第六章：大模型高效微调技术揭秘（下）

-   PEFT 基于重参数化（Reparametrization-based）训练方法

-   LoRA 低秩适配微调 （2021 Microsoft）

-   AdaLoRA 自适应权重矩阵微调 （2023 Microsoft, Princeton, Georgia Tech）

-   QLoRA 量化低秩适配微调 （2023 University of Washington）

-   UniPELT：大模型 PEFT 统一框架（2022）

-   （IA）^3：极简主义增量训练方法 （2022）

##  第七章：大模型高效微调工具 Hugging Face PEFT 入门与实战

-   Hugging Face PEFT 快速入门

-   PEFT 库是什么？

-   PEFT 与 Transformers 库集成

-   PEFT 核心类定义与功能说明

-   AutoPeftModels、PeftModel

-   PeftConfig

-   PeftType | TaskType

-   实战 PEFT 库 LoRA 模型微调

-   OpenAI Whisper 模型介绍

-   实战 LoRA 微调 Whisper-Large-v2 中文语音识别

##  第八章：大模型量化技术入门与实战

-   模型显存占用与量化技术简介

-   Transformers 原生支持的大模型量化算法

-   GPTQ：专为 GPT 设计的模型量化算法

-   AWQ：激活感知权重量化算法

-   BitsAndBytes（BnB） ：模型量化软件包

-   实战 Facebook OPT 模型量化

##  第九章：GLM 大模型家族与 ChatGLM3-6B 微调入门

-   智谱 GLM 大模型家族

-   基座模型 GLM-130B

-   扩展模型

-   联网检索能力 WebGLM

-   初探多模态 VisualGLM-6B

-   多模态预训练模型 CogVLM

-   代码生成模型 CodeGeex2

-   对话模型 ChatGLM 系列

-   ChatGLM3-6B 微调入门

-   实战 QLoRA 微调 ChatGLM3-6B

##  第十章：实战私有数据微调 ChatGLM3

-   实战构造私有的微调数据集

-   使用 ChatGPT 自动设计生成训练数据的 Prompt

-   合成数据： LangChain + GPT-3.5 Turbo

-   数据增强：提升训练数据多样性

-   提示工程：保持批量生成数据稳定性

-   实战私有数据微调 ChatGLM3

-   使用 QLoRA 小样本微调 ChatGLM3

-   ChatGLM3 微调前后效果对比

-   大模型训练过程分析与数据优化

##  第十一章：ChatGPT 大模型训练技术 RLHF

-   ChatGPT 大模型训练核心技术

-   阶段一：万亿级 Token 预训练语言模型

-   阶段二：有监督指令微调（SFT）语言模型

-   阶段三：使用 RLHF 实现人类价值观对齐（Alignment）

-   基于人类反馈的强化学习（RLHF）技术详解

-   步骤一：使用 SFT 微调预训练语言模型

-   步骤二：训练奖励模型（Reward Model）

-   步骤三：使用 PPO 优化微调语言模型

-   基于 AI 反馈的强化学习（RLAIF）技术

##  第十二章：混合专家模型（MoEs）技术揭秘

-   混合专家模型（Mixture-of-Experts, MoEs）技术发展简史

-   开山鼻祖：自适应局部专家混合（ Michael I. Jordan & Hinton, 1991）

-   多层次混合：深度 MoEs 中的表示学习（ Ilya, 2013）

-   稀疏门控：支持超大网络的 MoEs（Hinton & Jeff Dean, 2017）

-   MoEs 与 大模型结合后的技术发展

-   GShard：基于 MoE 探索巨型 Transformer 网络（Google, 2020）

-   GLaM：使用 MoE 扩展语言模型性能（Google, 2021）

-   Switch Transformer：使用稀疏技术实现万亿模型（Google, 2022）

-   MoEs 实例研究：Mixtral-8x7B-v0.1（Mistral AI, 2023）

##  第十三章：Meta AI 大模型 LLaMA

-   Meta LLaMA 1 大模型技术解读

-   基座模型系列：LLaMA1-7B（13B, 33B, 65B）

-   LLaMA 1 改进网络架构和预训练方法

-   LLaMA 1 衍生模型大家族

-   Meta LLaMA 2 大模型技术解读

-   基座模型系列： LLaMA2-7B（13B, 70B）

-   指令微调模型：LLaMA2-Chat

-   申请和获取 LLaMA 2 模型预训练权重

##  第十四章：实战 LLaMA2-7B 指令微调

-   大模型训练技术总结

-   以模型训练阶段分类：Pre-Training vs Fine-Tuning

-   以微调权重比例分类：FFT vs PEFT

-   以模型训练方法分类：Fine-Tuning vs Instruction-Tuning

-   以模型训练机制分类：SFT vs RLHF

-   实战 LLaMA2-7B 指令微调

-   指令微调格式：Alpaca Format

-   数据集：Databricks Dolly-15K

-   训练工具：HuggingFace TRL

-   上手训练 LLaMA2-7B 模型

##  第十五章：大模型分布式训练框架 Microsoft DeepSpeed

-   大模型分布式训练框架 DeepSpeed

-   预训练模型显存计算方法

-   Zero Redundancy Optimizer （ZeRO） 技术

-   DeepSpeed 框架和核心技术

-   分布式模型训练并行化技术对比

-   DeepSpeed 与 Transformers 集成训练大模型

-   DeepSpeed 实战

-   DeepSpeed 框架编译与安装

-   DeepSpeed ZeRO 配置详解

-   使用 DeepSpeed 单机多卡、分布式训练

-   实战 DeepSpeed ZeRO-2 和 ZeRO-3 单机单卡训练

-   DeepSpeed 创新模块：Inference、Compression & Science

##  第十六章：国产化适配 - 基于华为昇腾 910 微调 ChatGLM-6B

-   大模型算力设备与生态总结

-   蓝色星球的算力霸主：NVIDIA

-   厚积薄发的江湖大佬：Google

-   努力追赶的国产新秀：华为

-   华为昇腾全栈 AI 软硬件平台介绍

-   AI 开发平台：ModelArts

-   模型开发框架：MindSpore

-   异构计算架构：CANN

-   实战：华为 Ascend 910B 微调 ChatGLM-6B 模型

-   GLM 大模型家族介绍

-   GLM 系列模型发展历程：从 GLM-10B 到 GLM-4

-   GLM-4V：VisualGLM、CogVLM、CogAgent、GLM-4V 的技术变化

-   代码生成模型：CodeGeeX-3 及插件应用

-   图像生成模型 CogView-3

-   超拟人大模型 CharacterGLM

-   GLM-4 All Tools

-   GLM 模型部署微调实践

-   ChatGLM3-6B 开源介绍

-   ChatGLM3-6B 快速上手

-   模型微调

-   CogVLM 模型部署实践

-   CogVLM 开源模型介绍和体验

-   CogVLM 开源模型部署

## 实战项目


### **全量微调**

-   实战训练 BERT 模型：文本分类任务

-   数据集获取：下载 YelpReviewFull 数据集

-   数据预处理：清洗和准备

-   配置训练超参数

-   设置训练评估指标

-   介绍训练器基础知识

-   执行实战训练流程

-   保存训练好的模型

-   实战训练 BERT 模型：QA 任务

-   获取数据集：下载 SQuAD 数据集

-   数据预处理：准备和清洗

-   进行 Tokenizer 高级操作

-   使用 datasets.map 进行高级数据处理

-   微调模型设置

-   实例化训练器（Trainer）进行训练

-   进行模型评估

### **模型量化**

#### 实战 GPTQ 和 AWQ 模型量化（基于 OPT）

-   使用 AutoAWQ 量化模型

-   Transformers 兼容性配置

-   使用 GPU 加载量化模型

-   使用 GPTQ 量化模型

-   实测 GPU 显存占用峰值

-   检查量化模型正确性

-   使用自定义数据集量化模型

### **QLoRA**

#### 实战 LORA 微调 Whisper-Large-v2 中文语音识别

    -   全局参数设置

    -   数据准备

    -   下载数据集：训练、验证和评估集

    -   预处理数据：降采样、移除不必要字段等

    -   数据抽样（演示需要）

    -   应用数据集处理（`Dataset.map`）

    -   自定义语音数据处理器

### 模型准备

    -   加载和处理 `int8` 精度 Whisper-Large-v2 模型

    -   LoRA Adapter 参数配置

    -   实例化 PEFT Model：`peft_model = get_peft_model(model, config)`

#### 模型训练

    -   训练参数配置 Seq2SeqTrainingArguments

    -   实例化训练器 Seq2SeqTrainer

    -   训练模型

    -   保存模型

####  模型推理

    -   使用 `PeftModel` 加载 LoRA 微调后 Whisper 模型

    -   使用 `Pipeline API` 部署微调后 Whisper 实现中文语音识别任务

### 实战 QLoRA 微调 ChatGLM3-6B

#### 数据准备

    -   下载数据集

    -   设计 Tokenizer 函数处理样本（map、shuffle、flatten）

    -   自定义批量数据处理类 DataCollatorForChatGLM

#### 训练模型

    -   加载 ChatGLM3-6B 量化模型

    -   PEFT 量化模型预处理（prepare_model_for_kbit_training）

    -   QLoRA 适配器配置（TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING）

    -   微调训练超参数配置（TrainingArguments）

    -   开启训练（trainer.train)

    -   保存 QLoRA 模型（trainer.model.save_pretrained)

#### 模型推理

    -   加载 ChatGLM3-6B 基础模型

    -   加载 ChatGLM3-6B QLoRA 模型（PEFT Adapter）

    -   微调前后对比

### 实战私有数据微调 ChatGLM3

#### 实战构造私有的微调数据集

-   使用 ChatGPT 自动设计生成训练数据的 Prompt

-   合成数据： LangChain + GPT-3.5 Turbo

-   数据增强：提升训练数据多样性

-   提示工程：保持批量生成数据稳定性

###  实战私有数据 QLoRA 微调 ChatGLM3

-   全局参数设置
-   数据处理
-   模型准备和 QLoRA 训练
-   生成带有 epoch 和 timestamp 的模型文件
-   ChatGLM3 微调前后效果对比

-   大模型训练过程分析与数据优化

### **指令微调**

####  实战 LLaMA2-7B 指令微调

-   使用 Dolly-15K 数据集，以 Alpaca 指令风格生成训练数据

-   以 4-bit（NF4）量化精度加载 `LLaMA2-7B` 模型

-   使用 QLoRA 以 `bf16` 混合精度训练模型

-   使用 `HuggingFace TRL` 的 `SFTTrainer` 实现监督指令微调

-   使用 Flash Attention 快速注意力机制加速训练（需硬件支持）

### **分布式训练**

#### 实战 DeepSpeed ZeRO-2 和 ZeRO-3 训练

-   DeepSpeed 框架编译与安装

-   DeepSpeed ZeRO 配置详解

-   ZeRO-2 单机单卡训练 T5-Large

-   ZeRO-3 单机单卡训练 T5-Large

-   ZeRO-3 单机单卡训练 T5-3B

-   ZeRO-2 单机多卡训练 T5-3B