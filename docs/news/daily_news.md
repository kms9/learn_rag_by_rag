# 每日更新

## 2024-09-03

---
### 一款构建 Agent Memory 的工具：Memary

#### Memary 是一个自主代理记忆层，模拟人类记忆，让代理能够随着时间的推移学习和改进。与 Mem0 不同的是，它是一个记忆层框架，可以与各类型代理集成，提供更灵活的记忆生成、存储和检索功能。

#### 特点：
- 1、自动记忆生成：代理在与环境交互时，其记忆能够自动更新。
- 2、记忆模块：使用记忆流和实体知识库来分别跟踪用户接触过的概念的广度和深度，提供分析并展示在仪表板中。
- 3、知识图谱集成：通过 Neo4j 图数据库存储和管理知识，使用 Llama Index 和 Perplexity 进行知识注入和查询。
- 4、多模型支持：支持基于 Ollama 的本地 LLM 或 OpenAI 提供的模型。
- 5、支持自定义工具添加、记忆检索分析。

#### 链接地址：https://github.com/kingjulio8238/memary
---

---
### 一款构建 Agent Memory 的开源工具：RedCache-ai

#### 灵活的内存框架，支持自定义存储方式、数据格式和检索方式，可以满足不同应用场景。未来将提供 AI 智能体功能，可以构建更多复杂的应用。

#### 特点：
- 1、易使用：提供简单的 API，方便用户存储、检索、搜索、更新和删除内存。
- 2、多种存储方式：支持磁盘存储和 SQLite 数据库，可以根据需要选择合适的存储方式。
- 3、与 LLM 集成：目前支持 OpenAI，说未来会支持更多，像 Llama、Mixtral、Claude 等。

#### 链接地址：https://github.com/chisasaw/redcache-ai

---

---
### 基于 Meta LLama 3 70B 的开源 AI 搜索引擎项目：OpenPerPlex

#### 使用 Cohere 和 semantic-chunkers 库进行语义分块，使用 JINA API 对结果进行重新排序，通过 http://serper.dev 集成 Google 搜索，使用 Groq 作为推理引擎，LLM 用 Llama 3 70B。

#### 特点：
- 1、多库集成：整合 Cohere、semantic-chunkers 库、JINA API、Groq 等多种工具。
- 2、集成 Google 搜索：通过特定网址集成 Google 搜索。

#### 链接地址：https://github.com/YassKhazzan/openperplex_backend_os

### 一款自动漏洞修复工具：AI AppSec 工程师 ZeroPath

#### 可以自动检测和修复安全漏洞，可与现有 SAST 工具集成，能自动生成漏洞补丁并发出拉取请求进行修复。

#### 特点：
- 1、自动检测漏洞：分析代码识别潜在安全漏洞。
- 2、低误报率：过滤约 95%的误报。
- 3、自动生成补丁：为漏洞自动生成修复代码。
- 4、自然语言交互：支持用自然语言交互，比如提出修改要求。

#### 链接地址：https://zeropath.com

---
### 一款性价比极高的 AI 知识图谱构建工具：Triplex

#### 性能媲美 GPT-4o，成本是 GPT-4o 的十分之一。

#### 特点：
- 1、支持从文本中提取实体、关系和实体类型。
- 2、在准确性和效率方面超过 GPT-4o，同时成本更低。
- 3、模型尺寸更小，无需 few-shot 上下文。
- 4、与 R2R RAG 引擎和 Neo4J 集成，支持本地知识图谱构建。

#### 链接地址：
- 博客：https://sciphi.ai/blog/triplex
- 构建指南：https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph…
- github：https://github.com/SciPhi-AI/R2R
- 在线体验：https://kg.sciphi.ai
---

---
### 一款用于增强 LLM 在 RAG 任务中能力的框架：RAG Foundry

#### 英特尔发布的，支持数据创建、训练、推理和评估。

#### 特点：
- 1、可定制性：支持自定义各个模块，使用不同的检索器、提示器、评估指标等，满足不同任务和数据集的特定需求。
- 2、模块化设计：分为数据创建、训练、推理和评估四个独立的模块，可以灵活选择和组合，构建适合自己需求的 RAG 系统。
- 3、端到端实验环境：提供了从数据准备到模型训练和评估的一套流程，方便进行快速原型设计和实验，比较不同 RAG 性能。
- 4、支持多种 RAG 技术：链式思维、负样本训练等。
- 5、RAG 评估：提供了丰富的评估指标，检索准确性和生成质量等。

#### github：https://github.com/IntelLabs/RAGFoundry
#### 论文：https://arxiv.org/pdf/2408.02545
---

---
### 基于 LangGraph、FastAPI 和 Streamlit 构建 AI Agent 的完整工具包：agent-service-toolkit

#### 简介：提供从 Agent 定义到用户界面的完整架构，简化了使用 LangGraph 构建项目的过程。

#### 特点：
- 1、提供完整的 AI agent 服务架构
- 2、支持定制和扩展
- 3、具有用户友好的界面和反馈机制
- 4、支持 Docker 部署

#### 链接地址：https://github.com/JoshuaC215/agent-service-toolkit
---


---
### 一款多智能体服务系统：multi-agent-concierge

#### multi-agent-concierge 是基于 LLaMA 的多智能体对话系统，包含多个 agent，每个 agent 具备特定技能和知识，协同完成任务。

#### 特点：
- 1、多智能体协同，支持所 agent 并行对话。
- 2、支持通过自然语言指令创建不同的工作流程。
- 3、使用“监管”代理，管理跟踪每个 agent 的执行状态，确保整个工作流程顺利进行。
- 4、支持扩展。

#### 链接地址：https://github.com/run-llama/multi-agent-concierge
---

---
### 一个简化 RAG 系统构建的工具：Easy-RAG

#### Easy-RAG 是一个支持知识图谱实时提取解析和向量数据库应用的工具，提供了易于使用、模块化且可扩展的框架，实现 RAG 应用的快速构建。

#### 特点：
- 1、Easy-RAG 将 RAG 系统分成多个模块，比如数据预处理、索引构建、检索和生成，开发者可以根据需要选择和组合不同的模块。
- 2、支持多种预训练模型，BERT、RoBERTa 和 BART 等。
- 3、可扩展，可以根据需求添加新功能和模块。

#### 链接地址：https://github.com/yuntianhe2014/Easy-RAG
---


---
### 一个基于知识图谱的智能问答系统：fact-finder

#### 利用 LLM 和 Neo4j 数据库实现自动化查询与回答，实现了从用户问题到自然语言答案的自动化转换过程。

#### 特点：
- 1、基于知识图谱：使用 Neo4j 数据库存储和管理知识图谱，并利用它来回答问题。
- 2、利用语言模型：使用语言模型将问题转化为 Cypher 查询语句，并根据查询结果生成自然语言答案。
- 3、自动化查询与回答：将问题到自然语言答案的整个过程自动化，无需手动编写查询语句。

#### 链接地址：https://github.com/chrschy/fact-finder
---

---
### 一个 RAG 的集合库：RAG_Techniques

#### 涵盖了从基础 RAG 到复杂任务处理的多种方法，有详细文档、实现指南和示例，对需要深入了解 RAG 技术的开发者来说是个不错的选择。

#### 特点：
- 1、包含多种技术，如上下文丰富技术、多方面过滤、融合检索、智能重新排序、查询转换、分层索引、假设性问题 (HyDE 方法)、自 RAG 等。

#### 链接地址：https://github.com/NirDiamant/RAG_Techniques
---

---
### 一款开源的科学研究助手：OpenResearcher

#### 基于 RAG 技术，通过访问 arXiv 数据集，能理解用户问题并从科学文献中找到最相关答案，总结最新研究成果，在准确性、丰富性和相关性方面性能出色，媲美 Perplexity。

#### 特点：
- 1、查找并总结论文内容，比较不同论文观点，提供相关研究领域的其他资源。
- 2、支持多种 LLM，提供 Web 界面。

#### 链接地址：https://github.com/GAIR-NLP/OpenResearcher
---

---
### 一款开源 AI 搜索引擎：Sensei Search

#### Sensei，开源的、类似 perplexity 的 AI 驱动搜索引擎，利用 LLM 提供智能搜索和回答。支持本地和云端部署，适用于需要高效搜索和问答功能的各种应用场景。

#### 特点：
- 1、开源
- 2、AI 驱动
- 3、支持本地和云端部署

#### 链接地址：https://github.com/jjleng/sensei
---

---
### 一款工程师的 AI 助手：MLE-agent

#### MLE-agent 集成了 Arxiv 和 Papers with Code 平台，可以提供项目步骤计划建议，支持 OpenAI、Ollama 等。

#### 特点：
- 1、自动创建基线模型并根据用户需求规划项目步骤。
- 2、具备智能调试能力，可快速定位和解决代码问题。
- 3、集成 Arxiv、Papers with Code 等平台，获取最新研究成果和最佳实践，以保持其项目的前沿性。
- 4、集成了 AI/ML 功能和 MLOps，提供无缝工作流程，简化了从研究到生产的过程。
- 5、提供交互式聊天界面，支持用户以更自然的方式与工具交互。

#### 链接地址：https://github.com/MLSysOps/MLE-agent
---

---
### 一款基于 LLM 的网络爬虫工具：CyberScraper-2077

#### 集成了 OpenAI 和 Ollama，具有 AI 驱动、数据提取、友好界面、多格式支持、隐形模式、异步操作、智能解析和缓存等特点。

#### 特点：
- 1、AI 驱动：利用 AI 模型理解和解析网页内容
- 2、数据提取：可以从复杂或非结构化的数据源中提取信息
- 3、Streamlit 界面：用户友好的图形界面，易于操作
- 4、多格式支持：支持数据导出为 JSON、CSV、HTML、SQL 或 Excel 格式
- 5、隐形模式：内置隐形模式参数
- 6、异步操作：快速高效爬取数据
- 7、智能解析：将提取的内容结构化
- 8、缓存：使用 LRU 缓存和自定义字典实现内容和查询缓存，减少重复的 API 调用

#### 链接地址：https://github.com/itsOwen/CyberScraper-2077
---

---
### 一个由生成式 AI 驱动的超级个人助理项目：Quivr

#### 基于其 RAG 框架，构建用户的“第二大脑”，可以用于个人知识管理、数据整合、自动化任务、研究辅助、内容创作、翻译等等多用途。

#### 特点：
- 1、快速高效：以速度和效率为核心设计，可快速访问数据。
- 2、支持多种 LLM：GPT 3.5/4 turbo、Private、Anthropic、VertexAI、Ollama、Groq 等。
- 3、安全性：用户完全控制自己的数据。
- 4、文件兼容性：支持文本、Markdown、PDF、PowerPoint、Excel、CSV、Word、音频和视频文件。
- 5、公开/私有：可通过公共链接分享，或保持私有。
- 6、支持离线模式，支持分享。

#### 链接地址：https://github.com/QuivrHQ/quivr
---

---
### 一个开源的 AI 提示工程师项目：Ape

#### Ape 可以帮助用户更有效的与 AI 模型进行交互，通过设计和优化 prompt，引导 AI 生成更准确的输出，适用于需要与 AI 模型进行复杂交互的场景。

#### 特点：
- 1、支持 VS Code 扩展

#### 链接地址：https://github.com/weavel-ai/Ape
---

---
### 一个强大的基于 Ollama 的本地编程助手：Ollama-Engineer

#### Ollama-Engineer 是一个基于 Ollama 的编程助手，功能类似于 Claude Engineer，可以本地体验 Llama 3.1、Mistral Nemo 等的编程能力。支持多种语言编写、调试和改进代码，根据指令生成和应用代码编辑等，还支持项目管理、文件操作、网络搜索等。为了安全起见作者删除了代码执行工具。

#### 特点：
- 1、类似 Claude Engineer 的功能
- 2、本地体验多种模型编程能力
- 3、支持多种语言编写等操作
- 4、支持项目管理等功能

#### 链接地址：https://github.com/Doriandarko/claude-engineer
---

---
### Claude Engineer 的更新

#### Claude Engineer 的更新十分强大，现已支持多个文件/文件夹的创建和编辑。现在可以利用 3.5 Sonnet，一次性创建 6 个文件和 5 个文件夹，构建一个功能完备的 Web 应用。

#### 特点：
- 1、支持多文件/文件夹创建编辑
- 2、可利用 3.5 Sonnet 创建文件和文件夹构建 Web 应用

#### 链接地址：https://github.com/Doriandarko/claude-engineer
---

---
### 一款基于 RAG 的文档交互工具：kotaemon

#### 适用于文档问答、文档摘要、内容生成等场景。

#### 特点：
- 1、RAG：基于 RAG 能从文档中检索信息并生成答案。
- 2、多模型支持：包括 OpenAI、Azure OpenAI、Cohere，及本地模型等。
- 3、用户界面：提供了一个功能丰富、可定制的用户界面，可以轻松与文档交互。
- 4、可定制：可根据需求调整设置，包括检索和生成过程的配置等。
- 5、多模态支持：支持对包含图表和表格的多模态文档进行问答。
- 6、复杂问题处理：支持复杂推理方法，比如问题分解和基于代理的推理等。

#### github：https://github.com/Cinnamon/kotaemon

---
### 一个比较有意思的多代理智能系统项目：Co-STORM

#### Co-STORM 以讨论的方式进行复杂的信息搜索和学习。可以组成一个 agent 团队，围绕某个话题进行讨论，用户可以提出问题、分享观点或引导对话的方向，就像真实环境中头脑风暴一样，一旦满意，可以请求一份完整的带引文的报告。

#### 特点：
- 1、多代理对话：系统由多个 AI 代理组成，其可以模拟不同角色和观点，进行协作对话。
- 2、用户参与：用户可以观察代理间的对话，也可以在任何时候加入对话，提出问题或分享见解。
- 3、动态思维导图：使用动态思维导图跟踪对话内容和组织信息，使信息更加易于理解和记忆。
- 4、无需训练：可以直接使用，无需针对特定任务进行训练或调整。
- 5、生成报告：对话结束后，系统可以生成一份详细的报告，总结对话内容和收集的信息。
- 6、定制化：用户可以根据自己的需求和兴趣定制对话的内容和方向。
- 7、支持复杂查询：支持处理复杂的信息检索任务，比如学术研究、市场分析和决策制定等。

#### 链接地址：https://arxiv.org/pdf/2408.15232
---

---
### 一个基于命令行的 AI 编程助手：programmer

#### 具备自我学习和改进能力，能够直接访问机器运行命令，读写文件，但无安全检查。

#### 特点：

- 1、命令行界面：可以通过命令行与 AI 交互。
- 2、自动执行任务：能够自动执行如结束进程、分析图片内容、编写函数和单元测试等任务。

#### 链接地址：https://github.com/wandb/programmer
---

### 一款基于 AI 快速构建 web 应用的工具：GPT Engineer

#### GPT Engineer 支持使用自然语言快速构建 Web 应用，目前主要构建前端应用，如果与其他工具配合，可以构建全栈应用。

#### 特点：
- 1、基于 AI 生成代码：支持通过自然语言交互生成代码。
- 2、与 GitHub 同步：支持与 GitHub 同步，方便管理代码，进行版本控制。
- 3、支持一键部署到云平台。

#### 链接地址：https://gptengineer.app


### 一款AI 代码编辑器：Melty

####  Melty可以理解代码，协作完成任务，比如代码重构、创建 Web应用、导航大型代码库，甚至可以自动编写提交信息

####  特点：
- 1、AI 辅助: 基于AI更快速、更准确的编写代码
- 2、全流程集成: 支持与编译器、终端、调试器以及其他工具（Linear、 GitHub）集成
- 3、智能学习: Melty可以学习代码库，并根据用户习惯进行调整

#### github：https://github.com/meltylabs/melty



##  2024-07-05 
###   1. GraphRAG

- GraphRAG：负责任人工智能常见问题解答

什么是GraphRAG？

GraphRAG 是一种基于人工智能的内容解释和搜索能力。利用LLMs，它解析数据以创建知识图谱，并回答用户关于用户提供的私有数据集的问题。

介绍 
重磅 - 微软官宣正式在GitHub开源GraphRAG
实操 https://www.youtube.com/watch?v=f7Puiilv5Tw
小提示 随着文本量增加消耗token急剧增加   验证请使用本地模型

###   2. ai agent workflows
#### 吴恩达 为什么ai agent workflows很重要 
-    https://www.youtube.com/watch?v=sal78ACtGTc&ab_channel=SequoiaCapital
#### 一个ai agent workflow的实践
 -    https://twitter.com/AndrewYNg/status/1800582171259982289
#### 对吴恩达 workflow 概念产品化的思考 
 -   https://mp.weixin.qq.com/s/lk0h9ZtiR0BKlIoW2Kk8zw

###   3. DSPy
-     1. DSPy 是一个极好的 LLM 框架，它引入了一个自动编译器，教 LM 如何在程序中执行声明性步骤。具体来说，DSPy 编译器将在内部跟踪您的程序，然后为大型 LM (或为小型 LM 训练自动微调)制作高质量的提示

langchain中目前已经集成进了dspy

Feedback on 0.2 docs · langchain-ai/langchain · Discussion #21716

GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models

###   4. NL2SQL对比
- 1. Data-Copilot、Chat2DB、Vanna Text2SQL

##  2024-03-25
###   1. 首位完全自主的人工智能软件工程师德文•人工智能(Devin AI)的问世    https://devinai.ai/
#### 开源实现: 
#####    1. gpt-pilot
-    一个AI开发者伴侣，可以从0开始构建应用程序，可以自己编写代码、配置开发环境、管理开发任务、调试代码。 随时和它聊天提问，帮你解决开发难题。
  GitHub - Pythagora-io/gpt-pilot: The first real AI developer

#####    2.Devika
-     一个代理人工智能软件工程师，能够理解高层次的人工指令，将它们分解成几个步骤，研究相关信息，并编写代码来实现给定的目标。Devika 的目标是成为一个有竞争力的开放源码替代 Devin 的认知人工智能 
-    https://github.com/stitionai/devika
    
###   2.  替换搜索的专业方向ai助手
-     1. https://www.perplexity.ai/
-     2. https://www.phind.com/
-     3. https://devv.ai/
-     4. https://you.com/
-     5. https://search.tiangong.cn/

###   3. 目前一些rag知识库  
####       1. 可独立本地部署:
 -      1.  fastgpt
 -      2. dify.ai
 -      3. Chatollam
 -      4.  langflow
 -      5. Flowise
 -      6. Langchain-Chatchat 
 -      7. GitHub - casibase/casibase: ⚡️Open-source AI LangChain-like RAG (Retrieval-Augmented Generation) kno
 -      8. QAnything
####    2. 在线的
 -     1. Coze
 -     2. Chato
-    3. 基于langchain, llamaindex自己开发
    
## 2024-03-14
###  1. 沃顿商学院给教师和学生的提示词库
-     1. https://mp.weixin.qq.com/s/DgNh5cLEmyQwAQS52Ddwdw

##  21 款 AI 搜索引擎项目汇总

### Morphic

#### 生成式 UI：利用 Vercel AI SDK 和 OpenAI 模型，UI 上突破了纯文本和 markdown 限制，可以提供图片和链接资源。搜索功能：集成了 Tavily AI 和 Serper 的搜索 API，以及 Jina AI 阅读器 API，增强了搜索能力。

#### 特点：
- 1、生成式 UI 提供丰富内容
- 2、集成多种 API 增强搜索能力

#### github：https://github.com/miurla/morphic/tree/main
#### 网站：https://morphic.sh
---
### Farfalle

#### 克隆版 Perplexity，支持运行本地大模型 llama3、gemma、mistral、phi3；也支持云端模型 Groq/Llama3、OpenAI/gpt4-o。

#### 特点：
- 1、支持多种模型运行
- 2、既有本地模型也有云端模型

#### github：https://github.com/rashadphz/farfalle…
#### 网站：https://farfalle.dev

### Perplexica

#### 支持 ollama 部署运行本地模型，如 Llama-3、Mistral 和 Phi-3 等 Ollama 支持的大模型。提供两种模式：Copilot 模式和正常模式，Copilot 模式通过生成不同的查询来查找更相关的互联网资源来增强搜索。聚焦模式：特殊模式可以更好地回答特定类型的问题，如写作助手模式、学术检索模式等。即时信息：使用 searxng 获取即时信息，结合 RAG 进行组合优化。

#### 特点：
- 1、支持本地模型部署
- 2、多种模式增强搜索
- 3、即时信息获取与优化

#### github：https://github.com/ItzCrazyKns/Perplexica…
---
### Search4all

#### 集成了 OpenAI、Groq、Claude 等。原生搜索引擎集成了包括 Google、Bing、DuckDuckGo 和 SearXNG Search。用户界面可定制化。

#### 特点：
- 1、集成多个知名模型
- 2、原生搜索引擎丰富
- 3、用户界面可定制

#### github：https://github.com/fatwang2/search4all?tab=readme-ov-file…
#### 网站：https://search2ai.one
---
### MiniSearch

#### 无跟踪、无广告、无数据收集。界面简约直观，支持桌面和移动端。设置为默认搜索引擎，在浏览器地址栏中搜索。

#### 特点：
- 1、无跟踪等优点
- 2、简约直观界面
- 3、多端支持可设默认

#### github：https://github.com/lucaong/minisearch…
#### Demo：https://felladrin-minisearch.hf.space
---
### Lepton Search（贾扬清的 Search with Lepton）

#### 用少于 500 行代码构建自己的会话式搜索引擎。Lepton Search 使用 MistralAI 开源的 Mixtral-8x7b 作为支撑模型，运行在 LeptonAI 的 playground 托管平台上，吞吐量高达 200 tokens / 秒。内置支持大语言模型、搜索引擎、自定义 UI 界面，搜索结果可共享、缓存。

#### 特点：
- 1、代码量少可构建会话式搜索
- 2、高吞吐量
- 3、多种内置支持及可共享缓存结果

#### 网站：https://search.lepton.run
#### github：https://github.com/leptonai/search_with_lepton…
---
### llm-answer-engine

#### 使用 Next.js、Groq、Mixtral、Langchain、OpenAI、Brave 和 Serper 构建的 Perplexity 风格的回答引擎。

#### 特点：
- 1、多种技术构建
- 2、Perplexity 风格

#### github：https://github.com/developersdigest/llm-answer-engine…
#### 网站：https://developersdigest.tech
---
### LLocalSearch

#### 无需外部 API，数据处理完全本地化。支持在低配置硬件上运行使用。界面支持浅色和深色主题，兼容 PC 端及移动端。支持 Docker Compose 快速简单的部署方式。

#### 特点：
- 1、数据处理本地化
- 2、低配置硬件可用
- 3、多主题兼容多端可快速部署

#### github：https://github.com/nilsherzig/LLocalSearch…
---
### search_with_ai

#### 内置主流 LLM 接口 OpenAI、Google、通译千问、百度文心一言、Lepton、DeepSeek。内置搜索引擎 Bing、Sogou、Google、SearXNG（免费开源）。支持搜索引擎切换、AI 模型切换。支持基于 Ollama 的本地模型。

#### 特点：
- 1、内置多种接口和搜索引擎
- 2、可切换搜索引擎和模型
- 3、支持本地模型

#### github：https://github.com/yokingma/search_with_ai?tab=readme-ov-file…
---
### Perplexity

#### 一款知名的 AI 搜索引擎。

#### 特点：
- 1、具有较高的知名度
- 2、功能强大

#### 链接地址：https://perplexity.ai
---
### labs.perplexity.ai

#### 与 Perplexity 相关的项目。

#### 特点：
- 1、关联 Perplexity
- 2、可能有特定用途

#### 链接地址：https://labs.perplexity.ai
---
### 秘塔 AI 搜索（Metaso AI Search）

#### 秘塔推出的 AI 搜索工具。

#### 特点：
- 1、由秘塔开发
- 2、具有独特功能

#### 链接地址：https://metaso.cn
---
### Miku

#### 一款 AI 搜索产品。

#### 特点：
- 1、功能待探索
- 2、有一定特色

#### 链接地址：https://hellomiku.com
---
### Globe Explorer

#### 一款 AI 搜索工具。

#### 特点：
- 1、功能未知
- 2、可能有特定优势

#### 链接地址：https://explorer.globe.engineer
---
### 360AI 搜索

#### 360 推出的 AI 搜索服务。

#### 特点：
- 1、360 出品
- 2、有自身特点

#### 链接地址：https://so.360.com
---
### 天工 AI 搜索

#### 一款 AI 搜索产品。

#### 特点：
- 1、功能待了解
- 2、有独特之处

#### 链接地址：http://tiangong.cn
---
### Flowith

#### 一款 AI 搜索应用。

#### 特点：
- 1、特色未知
- 2、可能有特定用途

#### 链接地址：https://flo.ing/login
---
### 简单搜索 App (百度)

#### 百度推出的简单搜索应用。

#### 特点：
- 1、百度出品
- 2、简约风格

#### 链接地址：http://secr.baidu.com
---
### iAsk

#### 一款 AI 搜索工具。

#### 特点：
- 1、功能待发现
- 2、有一定特色

#### 链接地址：http://iask.ai
---
### You.com

#### 一款 AI 搜索网站。

#### 特点：
- 1、功能多样
- 2、有独特优势

#### 链接地址：http://you.com
---
### Bing Copilot

#### Bing 的辅助工具。

#### 特点：
- 1、与 Bing 结合
- 2、功能强大

#### 链接地址：http://bing.com/chat
---