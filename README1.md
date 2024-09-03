# RAG

## RAG-检索增强生成大纲

### LLM的局限性

### RAG概念

### RAG类比

### 为什么会用到RAG

## prompt

## RAG系统工作流程图解

###  

## RAG系统搭建流程

### 1. 加载文本与切割

### 2. 灌库构建检索引擎

### 3. 封装检索接口

### 4. 构建调用流程

## 关键字检索的局限性

## RAG进阶

### 在检索中:文本分割的粒度

### 在检索后:rank排序

## [向量检索](vector-seach.md)

### 1. 基本概念

- 什么是向量检索

	- 向量（embedding）的定义

	- 向量检索的基本原理

- 向量检索的重要性

	- 数据增长趋势

	- 非结构化数据处理的挑战

	- 向量检索的优势

### 2. 向量检索的应用场景

- 传统应用场景

	- 图片搜索

	- 音乐识别

- 结合大模型的应用场景

	- 大模型推理

	- 检索增强生成（RAG）

	- 跨模态应用

### 3. 向量检索的技术实现

- 向量索引

	- 实现方法一: 可扩展的最近邻搜索kNN

		- 效率非常低

		- 暴力检索

		- 检索质量高

	- 实现方式二: 近似最近邻搜索ANN

		- 倒排文件索引

			- 方法: IVF

			- 问题: 内存占用太大

				- 1. 保存每个向量的坐标

					- 解决: 量化 (Quantization) 有损压缩

						- 比如: 将聚类中心里面每一个向量都用聚类中心的向量来表示

				- 2. 维护聚类中心和每个向量的聚类中心索引

					- 问题: 高维坐标中的维度灾难问题

						- 维度灾难: 维度越高, 会需要更多的聚类中心点, 占用更多内存

							- 解决: Product Quantization (PQ) 量化乘积

								- 加快搜索速度

								- 减少内存开销

								- 降低搜索质量

		- 基于树的索引

			- 方法: Annoy (Approximate Nearest Neighbours Oh Yeah)

				- 特点: 低维数据表现好, 高维数据准确度不高

		- 基于图的索引

			- Hierarchical Navigable Small Worlds (HNSW) 以空间换时间

				- 类似跳表算法

				- 分层格式

					- 低层: 准确搜索

					- 高层: 快速搜索

				- 效果

					- 搜索质量高

					- 搜索速度快

					- 内存开销大

		- 图/树结合

			- 近邻图 (Neighborhood Graph and Trees, NGT) 索引

		- 基于哈希的索引

			- Locality Sensitive Hashing (LSH) 局部敏感hash

				- 增大哈希碰撞概率

				- 高维空间解决方案

					- Random Projection for LSH 随机投影

- 向量生成

	- 技术

		- 算法

			- 神经网络模型

			- Embedding 算法

		- 库

			- word2vec

			- fastText

			- bert

	- 可用服务

		- openai

		- 豆包

		- 本地部署

- 向量度量

	- Similarity Measurement
相似性测量
计算向量在高维空间的距离

		- Euclidean Distance
欧几里得距离

		- Cosine Similarity
余弦相似度

		- Dot product Similarity
点积相似度

### 向量数据库

- 向量数据库的作用

	- 高效的相似度搜索

	- 处理非结构化数据

	- 支持大规模数据处理

	- 增强大模型的能力

	- 动态数据处理

- 内部使用向量数据库

- 搭建外部向量数据库服务

- 主流向量数据库与功能对比

	- AI原生向量数据库

		- Faiss

			- 支持GPU加速

		- Chroma

		- Milvus/Zilliz

		- Pinecone

		- Qdrant

		- Weaviate

		- Vespa

	- 传统数据库拓展

		- pgvector

		- Cassandra

		- sqlite-vss

	- 传统搜索引擎系统扩展

		- Elasticsearch

		- OpenSearch

- 和传统数据库对比

	- 传统数据库(关系型、NoSQL)

		- 数据结构:结构化数据(行、列,具有预定义的数据类型)

		- 查询语言:类SQL/SQL拓展

		- 常见种类

			- PostgreSQL(行列表数据)

			- MongoDB(JSON文档存储数据)

			- Neo4j(点、边、属性存储关系)

		- 算法

			- 搜索(BTree索引、倒排索引)、精确匹配、排序

	- 向量数据库

		- 数据类型:非结构化数据(以向量或数组形式存储)

		- 查询语言:特殊语法或自定义接口

		- 应用场景:搜索、推荐系统、大模型应用等

### 4. 向量检索的优化

- 算法层面优化

	- 减少距离计算次数

	- 降低距离计算难度

- 系统层面优化

	- CPU 优化策略

	- FPGA 优化策略

	- 近存储处理

### 5. 向量检索的局限性

- 数据维度的挑战

- 计算资源的消耗

- 实时性要求

### 向量模型的本地部署

- ollama

- LMStudio

- Xinference

- vllm

