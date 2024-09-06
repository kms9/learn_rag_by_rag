# 每日总结

## 2024-09-06

### [hnsw搜索原理](../chapter2/hnsw.md)


### 读论文 [LLM Agents Improve Semantic Code Search](https://arxiv.org/html/2408.11058v1#S2)

#### 使用llm agent提高代码的语意搜索 看完了这个论文  一句话总结 就是先基于自然语言基于llm生成一些代码 然后基于这些代码再向量化之后去做向量检索, 作为多路召回的一个信息源  来提高最终匹配到目标代码的准确性 混合rag的一个应用例子

### 讨论代码大模型和代码助手
#### 目前开源的可自定义外部llm的代码助手
##### 1. cody  
1. 本地workspace功能依赖基于对本地sourcegraph的调用
##### 2. aider
1. 主要有一篇文章 https://aider.chat/2023/10/22/repomap.html
2. 基于 `tree sitter`来提高代码的可检索性
##### 3. continue
1. 有完整的本地codebase实现 基于sqlite的全文检索和lancedb的本地向量检索 甚至在vscode的插件里面下载了一个向量模型 `all-MiniLM-L6-v2`
2. 内部也基于 `tree sitter` 算法对代码进行结构分析