# Search Code

## case1
1. 使用信息注入作为改进代码搜索的方法。这种用例背后的原因是添加重要细节，以减轻代码搜索应用程序的用户提示中存在的模糊性和模糊性。通过利用代理 LLM 模型和 RAG，我们的系统能够执行与提示和 github 存储库相关的互联网搜索
   
### 一句话总结 就是先基于自然语言基于llm生成一些代码 然后基于这些代码再向量化之后去做检索 来提高最终匹配到目标代码的准确性 