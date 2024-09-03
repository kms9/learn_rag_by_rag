# 企业级 Agents 开发实战营

##  Agent 项目技术价值

###   工具类 Agent：GitHub Sentinel 
-  与 ChatGPT 结伴调研和设计一款 Agent 实用产品
-  快速入门基础的爬虫原理与实战
-  使用 GPT-4 从零到一生成项目级代码
-  学会容器化部署 AI Agent 服务
-  扩展性：任意信息流订阅和总结服务 Agent

####   对话类 Agent：LanguageMentor
-  学习 Ollama 私有化大模型管理工具
-  掌握面向多轮对话的复杂提示工程设计
-  覆盖海外衣食住行等使用场景的产品规划
-  基于 LangChain v0.3 最新接口研发
-  基于 LLaMA 3 设计端到端的私有化 Agent 产品
-  扩展性：100+语种的语言教练 Agent
####   多模态 Agent：ChatPPT
-  与 ChatGPT 一起调研分析企业办公场景
-  支持语音、图像和文本等多模态需求输入
-  面向多模态输入的提示工程设计
-  基于 GLM-4 设计和研发私有化部署 Agent
-  结合ReAct 理论与 Agent 项目实战
-  PowerPoint 文件自动化生成 SDK
-  扩展性：企业自动化流程提效 Agent
-  

### Agent 项目业务价值与功能介绍
- GitHub Sentinel 是一款开源工具类 AI Agent，专为开发者和项目管理人员设计，能够定期（每日/每周）自动获取并汇总订阅的 GitHub 仓库最新动态。其主要功能包括订阅管理、更新获取、通知系统、报告生成。通过及时获取和推送最新的仓库更新，GitHub Sentinel 大大提高了团队协作效率和项目管理的便捷性，使用户能够更高效地跟踪项目进展，快速响应和处理变更，确保项目始终处于最新状态。

- LanguageMentor 是一款基于 LLaMA 3 的在线英语私人教师 Agent，提供词汇积累、语法学习、阅读理解和写作训练等基础教学功能，以及模拟真实场景的对话训练。通过个性化定制学习路径，满足初学者、中级和高级学员的需求，为用户提供高效、便捷的语言学习体验，极大提高英语学习效率。

- ChatPPT 是一个基于多模态 AI 技术的智能助手，旨在提升企业办公自动化流程的效率。它能够处理语音、图像和文本等多种输入形式，通过精确的提示工程和强大的自然语言处理能力，为用户生成高质量的 PowerPoint 演示文稿。ChatPPT 不仅简化了信息收集和内容创作过程，还通过自动化的报告生成和分析功能，帮助企业快速、准确地完成各类汇报和展示任务，从而显著提升工作效率和业务价值。

## 大模型基础：理论与技术的演进
####   初探大模型：起源与发展
-  预热篇：解码注意力机制（Attention）
-  变革里程碑：Transformer 的崛起
-  走向不同：GPT 与 Bert 的选择
####   GPT 模型家族：从始至今
-  从 GPT-1 到 GPT-3.5：一路的风云变幻
-  ChatGPT：赢在哪里
-  GPT-4：一个新的开始

## GPT 大模型使用最佳实践
####   如何提升 GPT 模型使用效率与质量
#####   AI 大模型提示工程最佳实践
-  文本创作与生成
-  文章摘要和总结
-  小说生成与内容监管
-  分步骤执行复杂任务
-  评估模型输出质量
-  构造训练标注数据
-  代码调试助手
-  Playground Chat：实践 GPT 大模型提示工程
🌟工具类 Agent 实战篇
#### 大模型时代的 Agent 开发方法论
#####   设计和研发一款 Agent 的团队分工
-  产品经理：定义产品愿景和战略，制定计划，协调团队，收集和分析用户反馈
-  数据科学家：进行数据处理和机器学习模型构建，优化和部署模型
-  软件工程师：开发和维护前后端代码，实现 API 和接口，确保系统性能和安全
-  设计师：设计用户界面和用户体验，创建视觉元素和品牌标识
-  运维工程师：配置和维护服务器，部署容器化应用，监控和优化系统性能
-  市场营销：制定和执行市场推广策略，创建宣传材料，进行市场活动
-  客户支持：回答用户问题，收集反馈，提供培训，维护用户满意度
####   与 ChatGPT 结对研发闭环的工作模式
-  市场调研和竞争分析：收集和分析市场数据，提供竞争对手分析报告
-  用户反馈分析：处理用户反馈，进行情感分析，总结问题和建议
-  内容创建：创建市场宣传材料、博客文章、社交媒体内容等
-  客户支持：提供自动化客户支持，回答常见问题并解决基本问题
-  文档和报告编写：编写技术文档、用户手册和产品报告
-  产品设计初期阶段：提供用户界面设计建议和初步原型
-  开发支持：提供代码示例，解决编程问题，进行代码审查
####   AI Agent 和 ReAct 理论简介
-  AI Agent 概述
-  AI Agent 的定义和示例
-  AI Agent 的应用场景
-  ReAct 理论和主动学习
-  ReAct 理论在 AI Agent 中的应用
开源哨兵（GitHub Sentinel）Agent 立项与概念验证
####   GitHub Sentinel 项目立项
-  市场分析：明星开源项目的投资与商业价值
-  需求调研：GitHub 开源项目进展报告
-  产品设计：GitHub Sentinel 项目功能概述
-  技术方案：基于 GPT-4 的 Agent 架构设计
-  综合评估：GitHub Sentinel 研发成本与竞品分析
####   GitHub Sentinel 概念验证
-  闭源大模型分析能力：ChatGPT 4o
-  开源大模型分析能力：LLaMA 3 8B
####   开发环境与基础框架搭建
-  安装和配置开发工具（VS Code，Python 等）
-  基本的大模型开发环境设置
-  Fork 并配置 GitHub 项目仓库
-  初始化项目（OpenAI + LangChain）
-  项目目录结构介绍
GitHub Sentinel Agent 分析报告功能设计与实现
####   订阅管理功能
-  设计订阅管理功能的 API 接口
-  实现添加、删除和查看订阅的基础功能
-  数据库选择与初步配置
####   项目进展报告生成功能
-  设计分析 Pull Request 的提示工程
-  设计分析 Issues List 的提示工程
-  设计对比同类 GitHub 开源项目的提示工程
####   测试与调试
-  测试订阅管理和报告生成功能
-  调试常见问题
GitHub Sentinel Agent 定期更新功能设计与实现
####   GitHub 项目数据获取功能
-  设计更新获取的 API 接口
-  使用爬虫获取 GitHub PR 和 Issue 数据
-  使用 GitHub API 获取项目 Commit 记录
####   定期更新功能
-  使用 ChatGPT 生成 Python 原生调度器代码
-  使用 ChatGPT 生成 Cron 调度脚本
-  实现每日和每周定时任务
####   通知系统设计与实现
-  与 ChatGPT 探讨通知系统的功能与架构设计
-  SMTP 协议与使用介绍
-  使用 ChatGPT 生成邮件通知功能模块代码
####   测试与调试
-  测试定期更新与通知功能
-  调试常见问题
GitHub Sentinel Agent 用户界面设计与实现
####   用户界面开发
-  借助 ChatGPT 规划设计用户界面功能
-  创建用户界面原型
-  与 ChatGPT 探讨前端框架与技术选型
-  使用 ChatGPT 生成前端代码
####   后端 API 集成
-  连接前端与后端 API
-  数据传输与处理
####   命令行界面
-  设计并实现基本的命令行工具
-  结合 API 实现命令行操作
####   测试与调试
-  测试前后端功能与数据传输稳定性
-  测试命令行工具
-  调试常见问题
GitHub Sentinel Agent 高级功能与容器化部署
####   高级功能
-  实现 PDF 报告生成功能
-  实现 Word 报告生成功能
-  实现 第三方服务推送（如企业微信）
####   容器化部署
-  使用 DockerFile 编写项目依赖环境
-  使用 Docker 打包项目
-  部署到公有云平台（如：华为云、AWS、Azure）
-  配置持续集成和持续部署（CI/CD）
####   项目总结
-  项目复盘与总结
-  讨论项目扩展与优化的可能性
💫对话式 Agent 实战篇
私有化大模型管理工具 Ollama 快速入门
####   Ollama 101
-  Ollama 项目概述
-  Ollama 核心功能
-  Ollama 典型使用场景
-  Ollama 与 LLaMA 有关系么？
-  Ollama 会取代 LangChain 吗？
####   快速上手 Ollama 命令行交互
-  使用 Ollama 运行 LLaMA 3 8B 大模型
-  使用 Ollama 运行 Mistral 7B 大模型
-  使用 Ollama 运行 Gemma 7B 大模型
####   快速上手 Ollama REST API 对话服务
-  私有化部署和管理 LLaMA 3 模型服务
-  客户端调用 LLaMA 3 API 服务
LangChain v0.3 技术生态与未来发展
####   LangChain 大模型应用开发框架
-  LangChain 框架概述
-  LangChain 基础概念与发展历程
-  LangChain v0.3 破坏性升级（Breaking Change）详解
-  LangChain 社区生态的未来规划
####   LangChain v0.3 技术栈详解
-  LangChain 项目：Chains, Agents, Retrieval Strategy
-  LangChain Community 项目：Model I/O, Retrieval, Agent Tooling
-  LangChian-Core 项目：LangChain Expression Language（LCEL）
-  LangSmith 项目：Playground, Debugging, Evaluation
-  LangGraph 项目：High-level API for Multi-actor Agents
-  LangServe 项目：Chains to REST API Service
####   LangChain 工具调用（Tool Calling）
-  Function Calling vs Tool Calling
-  支持 Tool Calling 的大模型
-  使用 @tool 装饰器定义工具并绑定模型
-  使用聊天模型实现工具调用
-  使用 ToolMessage 管理工具调用输出
-  使用 Few-shot Prompting 提升工具调用质量
使用 LangGraph 构建生产级 AI Agent
####   LangGraph 快速入门
-  LangGraph 概述与背景
-  将 AI 工作流定义为图
-  使用图结构来构建复杂的 AI Agent
####   LangGraph 核心设计    
-  状态（State）：共享数据结构
-  节点（Nodes）：编码代理逻辑的 Python 函数
-  边（Edges）：控制流程规则
####   LangGraph 节点与边的实现
-  使用 Python 函数定义节点
-  使用方法add_node和add_edge添加节点和边
-  条件边和固定边的实现
####   LangGraph 状态管理
-  状态模式与 reducers 的使用
-  使用 TypedDict 和 Pydantic 管理状态
-  使用共享状态设计 AI 工作流
####   LangGraph 持久化与记忆
-  多回合记忆与单回合记忆
-  检查点系统的实现
-  使用compile(checkpointer)方法实现持久化
####   LangGraph 线程与配置
-  线程：支持多会话和多用户
-  可配置值的管理
-  在多回合交互中的应用
####   实战：使用 LangGraph 构建 Chatbots
-  Customer Support：差旅出行秘书
-  Info Gathering：信息整理助理
LangChain 表达式语言（LCEL） 入门与实战
####   LangChain Expression Language（LCEL）快速入门
-  LCEL 是什么
-  LCEL 设计理念与优势
-  LCEL Runnable 协议设计与使用
-  LCEL vs LangChain 古典语法
-  LCEL 进阶使用：整合复杂逻辑的多链
####   实战 LCEL 开发与应用
-  最简易示例：Prompt + Chat Model + Output Parser
-  检索增强生成示例：RAG
-  AI辩论示例：Multiple Chains via Runnables
英语私教（LanguageMentor） Agent 市场调研与产研设计
####   在线语言学习市场调研与分析
-  现有服务模式
        ■ 一对一私教：提供个性化定制服务，根据学生具体需求和学习目标设计课程
        ■ 小班授课：通常为 2-5 人的小班授课，注重互动性和竞争性
        ■ 在线课程：通过 Zoom、Skype 等平台进行，灵活安排时间和地点，适合现代快节奏生活
        ■ 自学课程：预录制的视频课程和在线练习，学生可以根据自己的节奏学习
-  现有收费模式
        ■ 按小时收费：一对一私教课每小时收费在50-150美元不等，根据教师资质和经验而定
        ■ 按课程收费：一些私教提供完整的课程包，按课程收取费用，通常比按小时收费更优惠
        ■ 订阅模式：部分在线平台采用月度或年度订阅模式，提供不限次数的课程和资源访问
####   LanguageMentor 产品设计
-  需求总结：用户对私人语言辅导的需求和期望不断增加
-  核心功能：
        ■ 基础教学：涵盖词汇积累、语法学习、阅读理解和写作技巧等基础内容
        ■ 对话式训练：模拟真实场景的对话练习，提升学生的口语表达能力和听力理解能力
-  用户学习路径：
        ■ 初学者：注重词汇和基础语法的学习，通过简单对话练习提高自信心
        ■ 中级学员：结合复杂语法和高级词汇，进行更深入的阅读和写作训练
        ■ 高级学员：重点练习口语和听力，通过模拟真实场景的对话提升实战能力
-  课程设计：
        ■ 词汇积累：采用词根词缀法和常用词汇表，帮助学生高效记忆单词
        ■ 语法学习：通过系统的语法讲解和练习，夯实学生的语法基础
        ■ 阅读理解：提供不同难度的阅读材料，训练学生的阅读速度和理解能力
        ■ 写作技巧：指导学生如何进行段落和文章的结构化写作
####   LanguageMentor 技术方案
-  大模型选型：使用 Ollama 运行和管理 LLaMA 3 本地化模型
-  用户交互：基于 Gradio 的图形化聊天界面
-  Agent 逻辑设计：使用 LangChain 整合复杂的逻辑链
-  综合评估：LanguageMentor 研发成本与竞品分析
LanguageMentor Agent 基础教学功能设计与实现
####   开发环境与基础框架搭建
-  私有化大模型开发环境设置（LLaMA 3，Gemma等）
-  Fork 并配置 GitHub 项目仓库
-  初始化项目（LLaMA 3 + LangChain）
-  项目目录结构介绍
####   基础教学功能
-  词汇积累模块：设计词汇学习的提示工程
-  语法学习模块：设计语法学习的提示工程
-  阅读理解模块：设计阅读理解的提示工程
-  写作练习模块：设计写作练习的提示工程
####   测试与调试
-  测试各模块的基本功能
-  调试常见问题
LanguageMentor Agent 对话训练功能设计与实现
####   对话训练功能
-  日常对话：设计模拟日常对话的提示工程
-  特定场景对话：
        ■ 设计技术面试场景的 Agent 方案
        ■ 设计饭店点餐场景的 Agent 方案
        ■ 设计商务会议场景的 Agent 方案
####   高级对话功能
-  实现情感识别，调整对话内容的语气和难度
-  对话逻辑优化：提升多轮对话的连贯性和自然性
####   测试与调试
-  测试各模块的基本功能
-  调试常见问题
LanguageMentor Agent 容器化部署与发布
####   Docker 镜像创建与测试
-  编写 Dockerfile，定义应用的依赖环境和配置。
-  创建 Docker 镜像，确保应用环境的一致性和可移植性。
-  配置 Docker 容器的网络和存储，确保应用的稳定运行。
-  进行本地测试，确保容器内的应用功能正常。
####   应用部署
-  选择合适的云平台（如 AWS、Azure、GCP），部署 Docker 容器。
-  配置负载均衡和自动扩展，确保应用在高并发下的性能。
-  配置监控工具和日志系统，实时监控应用的运行状态并收集用户反馈。
多模态 Agent 实战篇
## ChatPPT 项目立项与概念验证
####   使用 ChatGPT 进行企业办公场景市场调研
-  企业办公的常见问题与痛点
-  办公自动化的需求和期望
-  企业办公场景的成功案例研究与分析
-  生成调研报告和分析结果
-  PowerPoint 使用场景研究
####   ChatPPT 产品设计与核心功能
-  多模态需求输入：图像、语音、文本
-  PowerPoint 内容生成与分页处理
-  多轮需求处理与版本迭代
-  PowerPoint 样式模板库
####   ChatPPT 架构设计与技术方案验证
-  语音输入集成模块
-  图像输入集成模块
-  核心需求理解与多轮输入整合模块
-  PowerPoint 不同场景主题的提示工程设计
-  PowerPoint 文件自动化生成模块
-  综合评估：ChatPPT 研发成本与竞品分析
## 实现文本到 PowerPoint 的 ChatPPT v1.0
####   PowerPoint 文件结构与自动生成
-  介绍 python-pptx 库
-  使用 Python 创建和修改 PowerPoint 文件
-  实现自动化 PowerPoint 生成
####   设计 PowerPoint 内容生成的提示工程
-  智谱开源 GLM 4 大模型介绍
-  GLM 4 模型的部署与测试
-  基于 GLM 4 的提示工程设计与测试
####   ChatPPT v1.0 研发实践
-  集成 GLM 4 提示工程和 PowerPoint 文件生成模块
-  基于 gradio 的图形化界面研发
-  测试与调试常见问题
集成语音输入的 ChatPPT v2.0
####   语音输入处理
-  自动语音识别（ASR）的工作原理
-  常用的 ASR 模型和技术
-  语音数据的预处理工作
####   语音输入需求集成
-  将 ASR 模块集成到 ChatPPT 中
-  处理和解释语音数据
-  创建基于语音输入的提示工程
####   ChatPPT v2.0 研发实践
-  实现并测试 ASR 模块
-  整合 ASR 模块到 ChatPPT 中
-  收集和准备语音数据集，进一步集成测试
-  测试与调试常见问题
## 集成图像输入与多模态提示工程设计 ChatPPT v3.0
####   图像输入处理
-  图像识别的工作原理
-  常用的图像识别模型和技术
-  图像数据的预处理和特征提取
####   图像输入需求集成
-  将图像识别模块集成到 ChatPPT 中
-  处理和解释图像数据
-  图像作为内容嵌入 PowerPoint
-  实现并测试图像识别模块
####   集成多模态输入的提示工程设计
-  综合多模态提示的设计方法
-  创建和测试多种类型的提示
-  为特定用例开发多模态输入提示
-  将多模态提示工程整合到 ChatPPT 中
####   ChatPPT v3.0 研发实践
-  整合图像识别模块到 ChatPPT 中
-  收集和准备图像数据集，进一步集成测试
-  测试系统的多模态输入功能
-  多模态需求整合的集成测试
## ChatPPT Agent 容器化部署与发布
####   Docker 镜像创建与测试
-  编写 Dockerfile，定义 ChatPPT 的依赖环境和配置
-  创建 Docker 镜像，确保应用环境的一致性和可移植性
-  配置 Docker 容器的网络和存储，确保应用的稳定运行
-  进行本地测试，确保容器内的 ChatPPT 功能正常
####   ChatPPT 应用部署
-  选择合适的云平台（如 AWS、Azure、GCP），部署 Docker 容器
-  配置负载均衡和自动扩展，确保 ChatPPT 在高并发下的性能
-  配置监控工具和日志系统，实时监控应用的运行状态并收集用户反馈
####   集成与测试
-  测试容器化部署的稳定性和性能
-  优化容器配置，提升应用的运行效率
项目简介
工具类 Agent -- GitHub Sentinel 
GitHub Sentinel 是一款开源工具类 AI Agent，专为开发者和项目管理人员设计，能够定期（每日/每周）自动获取并汇总订阅的 GitHub 仓库最新动态。其主要功能包括订阅管理、更新获取、通知系统、报告生成。通过及时获取和推送最新的仓库更新，GitHub Sentinel 大大提高了团队协作效率和项目管理的便捷性，使用户能够更高效地跟踪项目进展，快速响应和处理变更，确保项目始终处于最新状态。
####   技术价值
-  与 ChatGPT 结伴调研和设计一款 Agent 实用产品
-  快速入门基础的爬虫原理与实战
-  使用 GPT-4 从零到一生成项目级代码
-  学会容器化部署 AI Agent 服务
-  扩展性：任意信息流订阅和总结服务 Agent
对话式 Agent -- LanguageMentor
LanguageMentor 是一款基于 LLaMA 3 的在线英语私人教师 Agent，提供词汇积累、语法学习、阅读理解和写作训练等基础教学功能，以及模拟真实场景的对话训练。通过个性化定制学习路径，满足初学者、中级和高级学员的需求，为用户提供高效、便捷的语言学习体验，极大提高英语学习效率。
####   技术价值
-  学习 Ollama 私有化大模型管理工具
-  掌握面向多轮对话的复杂提示工程设计
-  覆盖海外衣食住行等使用场景的产品规划
-  基于 LangChain v0.3 最新接口研发
-  基于 LLaMA 3 设计端到端的私有化 Agent 产品
-  扩展性：100+语种的语言教练 Agent
多模态 Agent -- ChatPPT
ChatPPT 是一个基于多模态 AI 技术的智能助手，旨在提升企业办公自动化流程的效率。它能够处理语音、图像和文本等多种输入形式，通过精确的提示工程和强大的自然语言处理能力，为用户生成高质量的 PowerPoint 演示文稿。ChatPPT 不仅简化了信息收集和内容创作过程，还通过自动化的报告生成和分析功能，帮助企业快速、准确地完成各类汇报和展示任务，从而显著提升工作效率和业务价值。
####   技术价值
-  与 ChatGPT 一起调研分析企业办公场景
-  支持语音、图像和文本等多模态需求输入
-  面向多模态输入的提示工程设计
-  基于 GLM-4 设计和研发私有化部署 Agent
-  结合ReAct 理论与 Agent 项目实战
-  PowerPoint 文件自动化生成 SDK
-  扩展性：企业自动化流程提效 Agent