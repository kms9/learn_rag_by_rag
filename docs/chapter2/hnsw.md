
# 像光速一样搜索——HNSW算法介绍

> 本文是一篇英文翻译转载文章，主要介绍了HNSW算法。
> URL Source: https://luxiangdong.com/2023/11/06/hnsw/
> 原文链接：[https://dataman-ai.medium.com/search-like-light-speed-1-hnsw-c5b0d4665926](https://dataman-ai.medium.com/search-like-light-speed-1-hnsw-c5b0d4665926)

我喜欢《玩具总动员》里的太空护林员“巴斯光年”，我喜欢他的口头禅“飞向无限！” 当我搜索信息时，我也享受找到正确信息的速度。这一切都是因为高速互联网和足够的带宽吗？不完全是！**事实上，近乎即时搜索结果的算法是至关重要的**。信息检索速度是计算机科学中的一个重要课题。随着文本、图像或音频数据的大型语言模型（大语言模型）的高维Embeddings，信息检索的速度是数据科学中的一个优先课题。

在这篇文章中，我将讨论:

*   NLP中的向量Embeddings
*   KNN (K-Nearest Neighbors)无法跟上速度
*   近似最近邻(ANN)感觉就像光速
*   快速搜索的初始算法
*   理解分层导航小世界图算法(HNSW)
*   代码示例:Embedding新闻文章
*   代码示例:FAISS用于HNSW搜索

本文及其后续系列解释了使巴斯光年的梦想成为可能的最先进算法。您将对这一领域及其应用的重要性有一个景观理解。您将有动手编码示例。我们开始吧。

**NLP中的向量Embeddings**

向量Embeddings是自然语言处理（NLP）中的一个基本概念，是单词、句子、文档、图像、音频或视频数据等对象的数字表示。这些Embeddings的目的是捕获它们所表示的对象的语义和上下文信息。

让我们首先描述一下单词Embeddings。2014年，一个突破性的想法Word2Vec(发音为“Word - to - Vector”)在自然语言处理中被提出，它将单词或短语转换或“嵌入”为数字的高维向量，称为单词Embeddings。这些词Embeddings捕捉词之间的语义和上下文关系，使机器能够理解和使用人类语言。图1显示了三维空间中的高维向量。“铁（iron）”这个词与“火药（gunpowder）”、“金属（metals）”和“钢（steel）”等词很接近，但与“有机（organic）”、“糖（sugar）”或“谷物（grain）”等不相关的词相去甚远。例如，猫和狗的概念可能很接近。

![Image 1: img](https://luxiangdong.com/images/hnsw/1.png)

图 (1)：文字Embeddings(图片来源:作者)

单词Embeddings可以实现单词的相似或不相似。这是一项了不起的创新。既然单词可以嵌入，为什么句子不能呢？这就是句子Embeddings的诞生。句子Embeddings捕获整个句子的语义和上下文信息，使机器能够理解和比较句子。生成句子Embeddings的常用方法包括Doc2Vec (Document-to-vector)。强大的基于llm的词Embeddings将成为NLP的标准，如BERT(来自Transformers的双向编码器表示)、ELMo(来自语言模型的Embeddings)、Llama(大型语言模型元AI，由Meta AI于2023年2月推出)，以及OpenAI的多个模型。

既然文本可以作为向量嵌入，为什么不能嵌入图像呢？这就产生了图像Embeddings。卷积神经网络(cnn)和视觉几何组(VGG)用于生成图像Embeddings。图像Embeddings使图像检索和分类成为可能。

既然图像可以作为矢量嵌入，为什么不能嵌入音频呢？你说得对！音频Embeddings可以捕获音频数据的声学特征，并可以进行音频分类和检索。视频Embeddings如何？它们捕获图像特征流用于视频分类。那么地理空间Embeddings呢？当然可以。纬度和经度坐标等地理空间数据可以嵌入到高维向量中，便于信息检索。

Embeddings使一切变得简单。如果你有一篇文章，需要找到类似的文章，你只需要计算你的文章的向量到其他文章的向量之间的距离。最近的向量就是你的搜索结果。我们可以用k近邻法（KNN），对吧？然而，速度是个问题。**KNN的搜索将使光年皱眉**。对于巴斯光年来说，完成一次简单的搜索需要…需要不知道多少年。研究的挑战不是最近的邻居在哪里，而是“如何”找到它们。

**k-最近邻(KNNs)无法跟上速度**

假设你有一本新书，你想在图书馆找到类似的书。k-最近邻（KNN）将浏览书架上的每一本书，并将它们从最相似到最不相似的顺序排列，以确定最相似的书。你有耐心做这么麻烦的工作吗？相反，人工神经网络对图书馆中的图书进行预排序和索引。要找到与你的新书相似的书，你所需要做的就是去正确的楼层，正确的区域，正确的通道找到相似的书。此外，**你通常不需要对前10本相似的书进行精确排名，比如100%、99%或95%的匹配度**。这就是**近似近邻（ANN）**的思想。

![Image 2: img](https://luxiangdong.com/images/hnsw/2.png)

让我们来了解一下为什么人工神经网络可以更有效地搜索。

**近似最近邻(ANN)感觉像光速**

ANN （Approximate Nearest Neighbors）对大数据进行预索引，方便快速搜索。在索引期间，构建数据结构以促进更快的查询。当您想为一个查询点找到近似的最近邻居时，您可以将该查询点提供给ANN算法。人工神经网络算法首先从数据集中识别一组可能接近查询点的候选数据点。使用预构建的数据结构选择候选对象。这一步骤大大减少了需要检查接近性的数据点的数量。在候选点被选中之前，ANN计算每个候选点与查询点之间的实际距离(如欧几里得距离、余弦相似度)。然后根据与查询点的距离/相似度对候选项进行排名。排名靠前的候选人作为近似近邻返回。在某些情况下，还可以设置距离阈值，只返回该阈值内的候选对象。人工神经网络背后的关键思想是，**为了显著降低计算成本，它牺牲了找到绝对最近邻的保证。这些算法的目标是在计算效率和准确性之间取得平衡**。

然而，在高维空间中，过去的实验表明ANN并不比KNN节省多少时间(见\[4\])。有几种创新的人工神经网络算法适用于高维空间。我将列出这些算法的字母表。您很快就会熟悉这些字母，并且可能更愿意在NLP社区中使用它们进行交流。让我们学习流行的最先进的算法。

**最先进的快速搜索算法**

这些不同的人工神经网络算法是不同的方法来形成**数据结构**，以实现有效的检索。有三种类型的算法：基于图的、基于哈希的和基于树的。

**基于图的算法创建数据的图**表示，其中每个数据点是一个节点，边表示数据点之间的接近性或相似性。最引人注目的是层次导航小世界图(HNSW)。

**基于哈希的算法**使用哈希函数将数据点映射到哈希码或桶。流行的算法包括:位置敏感哈希（LSH）、多索引哈希（MIH）和产品量化

**基于树的算法**将数据集划分为树状结构，以便快速搜索。流行的是kd树、球树和随机投影树（RP树）。对于低维空间（≤10），基于树的算法是非常有效的。

有几个流行的代码库:

1.  **Scikit-learn**：它的`NearestNeighbors`类提供了一个简单的接口，可以使用LSH等技术进行精确和近似的最近邻搜索。
2.  **Hnswlib**：它是HNSW的Python包装器。
3.  **FAISS**：该库支持多种ANN算法，包括HNSW, IVFADC(带ADC量化的倒置文件)和IVFPQ(带产品量化的倒置文件)。
4.  **Annoy** (Approximate Nearest Neighbors Oh Yeah)：Annoy是一个c++库，也提供了Python接口。
5.  **NMSLIB**(非度量空间库)：它是用c++编写的，并具有Python包装器。它可以执行HNSW、LSH、MIH或随机投影树等算法。

使用上述代码库，您可以超级快速地执行搜索查询。您还需要了解其他库的变体。这里我只提到其中的三个。第一个是[PyNNDescent](https://github.com/lmcinnes/pynndescent)。PyNNDescent是一个Python库，用于基于NN-descent的搜索算法，它是LSH的一个变体。第二个是[NearPy](http://pixelogik.github.io/NearPy/)。它支持多个距离度量和哈希族。第三个是[PyKDTree](https://github.com/storpipfugl/pykdtree)。PyKDTree是一个Python库，用于基于kd树的最近邻（KNN）搜索。虽然kd树更常用于精确搜索，但PyKDTree也可以通过一些启发式优化用于近似搜索。

此外，如果您询问哪些算法和库执行速度最好，您只需要了解[\*\*ANN- benchmark \*\*](https://github.com/erikbern/ann-benchmarks)库，专门为对人工神经网络搜索算法进行基准测试而设计。它提供了一个标准化的框架来评估算法，如[Annoy](https://github.com/spotify/annoy)， [FLANN](http://www.cs.ubc.ca/research/flann/)， [scikit-learn](http://scikit-learn.org/stable/modules/neighbors.html) (LSHForest, KDTree, BallTree)， [PANNS](https://github.com/ryanrhymes/panns)， [NearPy](http://pixelogik.github.io/NearPy/)， [KGraph](https://github.com/aaalgo/kgraph)， [NMSLIB(非度量空间库)](https://github.com/nmslib/nmslib)，[hnswlib](https://github.com/nmslib/hnsw)， [RPForest](https://github.com/lyst/rpforest)， [FAISS](https://github.com/facebookresearch/faiss)， [nndescent](https://github.com/brj0/nndescent)， [PyNNDescent](https://github.com/lmcinnes/pynndescent)等等。

这不仅仅是最近邻在哪里，而是如何有效地找到他们。让我们学习第一个算法——HNSW。**HNSW通常可以在几毫秒内从数百万个数据点中找到最近邻。**

**了解分层导航小世界图(HNSW)**

HNSW是一种用于在高维空间中进行高效人工神经网络搜索的数据结构和算法。它是**跳表**和**小世界图（SWG）**结构的扩展，可以有效地找到近似的最近邻。如果我们先学习跳表和小世界图，学习HNSW就会很简单。

跳表是一种数据结构，用于维护一组已排序的元素，并允许进行高效的搜索、插入和删除操作。它是由William Pugh在1989年发明的。图(2)显示了数字\[3、6、7、9、12、17、19、21、25、26\]的排序链表。假设我们想找到目标19。当值小于目标时，我们向右移动。需要6步才能找到它。

![Image 3: img](https://luxiangdong.com/images/hnsw/3.png)

图 (2): 排序链表

现在，如果列表的每个其他节点都有一个指向前面节点2的指针，如图3所示，可以将这些指针视为“高速公路”。数学规则是“当数值小于目标时向右移动”。需要4个步骤才能达到19。

![Image 4: img](https://luxiangdong.com/images/hnsw/4.png)

图 (3): 跳表，其指针指向后面两个节点

这些高速公路加快了搜索速度。我们可以增加更多。现在，如果列表中每三个其他节点都有一个指向前面第三个节点的指针，如图(4)所示，那么只需要3步就可以到达19。

你可能会问，如何选择这些点作为”高速公路“？它们可以是预先确定的或随机选择的。这些节点的随机选择是Small World和NHSW中数据构建的重要步骤，我将在后面介绍。

![Image 5: img](https://luxiangdong.com/images/hnsw/5.png)

图 (4): 跳表再升级，指向后面三个节点的指针

由跳表的思路延伸到Small World，我们来看看是怎么做的。

**由跳表的思路延伸到Small World**

小世界（small world）网络是一种特殊的网络，在这种网络中，你可以快速地联系到网络中的其他人或点。这有点像“凯文·培根的六度”(Six Degrees of Kevin Bacon)游戏，在这个游戏中，你可以通过一系列其他演员，在不到六个步骤的时间里，将任何演员与凯文·培根联系起来。

想象一下，你有一群朋友排成一个圆圈，如图5所示。每个朋友都与坐在他们旁边的人直接相连。我们称它为“原始圆”。

现在，这就是奇迹发生的地方。你可以随机选择将其中一些连接改变给圆圈中的其他人，就像图5中的红色连接线一样。这就像这些连接的“抢椅子”游戏。有人跳到另一把椅子上的几率用概率p表示。如果p很小，移动的人就不多，网络看起来就很像原来的圆圈。但如果p很大，很多人就会跳来跳去，网络就会变得有点混乱。当您选择正确的p值(不太小也不太大)时，红色连接是最优的。网络变成了一个小世界网络。你可以很快地从一个朋友转到另一个朋友(这就是“小世界”的特点)。

![Image 6: img](https://luxiangdong.com/images/hnsw/6.png)

图 (5): small-world网络

现在让我们学习从小世界网络到可导航小世界的过渡。

**从小世界到HNSW**

现在我们要扩展到高维空间。图中的每个节点都是一个高维向量。在高维空间中，搜索速度会变慢。这是不可避免的“维度的诅咒”。HNSW是一种高级数据结构，用于优化高维空间中的相似性搜索。

让我们看看HNSW如何构建图的层次结构。HNSW从图(6)中的第0层这样的基础图开始。它通常使用随机初始化数据点来构建。

![Image 7: img](https://luxiangdong.com/images/hnsw/7.png)

图 (6): HNSW

HNSW在层次结构中的基础层之上构造附加层。每个层将有更少的顶点和边的数量。可以把高层中的顶点看作是跳跃列表中访问“高速公路”的点。你也可以将这些顶点视为游戏“Six Degrees of Kevin Bacon”中的演员Kevin Bacon，其他顶点可以在不到6步的时间内连接到他。

一旦构建了上面的层次结构，数据点就被编入索引，并准备进行查询搜索。假设查询点是桃色数据点。为了找到一个近似最近的邻居，HNSW从入门级(第2层)开始，并通过层次结构向下遍历以找到最近的顶点。在遍历的每一步，算法检查从查询点到当前节点邻居的距离，然后选择距离最小的相邻节点作为下一个基本节点。查询点到最近邻居之间的距离是常用的度量，如欧几里得距离或余弦相似度。当满足某个停止条件(例如距离计算次数)时，搜索终止。

现在让我们看看HNSW是如何构建这些层的。

**HNSW如何构建数据结构?**

HNSW首先初始化一个空图作为数据结构的基础。该图表示一个接一个插入数据点的空间。HNSW将数据点组织成多层。每一层表示数据结构中不同级别的粒度。层数是预定义的，通常取决于数据的特征。

每个数据点随机分配到一个层。最高的一层用于最粗略的表示，随着层的向下移动，表示变得更精细。这个任务是用一个特定的概率分布来完成的，这个概率分布叫做指数衰减概率分布。这种分布使得数据点到达更高层的可能性大大降低。如果你还记得跳跃列表中随机选择的数据点作为“高速公路”，这里的一些数据点是随机选择到最高层的。在后面的代码示例中，我们将看到每层中的数据点数量，并且数据点的数量在更高层中呈指数级减少。

为了在每一层内有效地构建连接，HNSW使用贪婪搜索算法。它从顶层开始，试图将每个数据点连接到同一层内最近的邻居。一旦建立了一层中的连接，HNSW将使用连接点作为搜索的起点继续向下扩展到下一层。构建过程一直持续到处理完所有层，并且完全构建了数据结构。

让我们简单总结一下HNSW中数据结构的构造。让我也参考Malkov和Yashunin\[3\]中的符号，并在附录中解释HNSW算法。您可能会发现它们有助于更明确地理解HNSW的算法。HNSW声明一个空结构并逐个插入数据元素。它保持每个数据点每层最多有_M_个连接的属性，并且每个数据点的连接总数不超过最大值(_Mmax_)。在每一层中，HNSW找到与新数据点最近的K个邻居。然后，它根据距离更新候选数据点集和找到的最近邻居列表(_W_)。如果_W_中的数据点数量超过了动态候选列表(_ef_)的大小，则该函数从_W_中删除最远的数据点。

接下来，我将向您展示代码示例。该笔记本可通过[此链接](https://github.com/dataman-git/codes_for_articles/blob/master/HNSW.ipynb)获得。

**代码示例**

接下来，让我们使用库FAISS执行HNSW搜索。我将使用NLP中包含新闻文章的流行数据集。然后，我使用“SentenceTransformer”执行Embeddings。然后，我将向您展示如何使用HNSW通过查询搜索类似的文章。

**Data**

总检察长的新闻文章语料库由\[A.\]Gulli\]([http://groups.di.unipi.it/~gulli/AG\_corpus\_of\_news\_articles.html)，是一个从2000多个新闻来源收集100多万篇新闻文章的大型网站。Zhang、Zhao和LeCun在论文中构建了一个较小的集合，其中采样了“世界”、“体育”、“商业”和“科学”等新闻文章，并将其用作文本分类基准。这个数据集“ag\_news”已经成为一个经常使用的数据集，可以在Kaggle、PyTorch、Huggingface和Tensorflow中使用。让我们下载\[数据从Kaggle\](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)。训练样本和测试样本分别有12万篇和7600篇新闻文章。](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html/)%EF%BC%8C%E6%98%AF%E4%B8%80%E4%B8%AA%E4%BB%8E2000%E5%A4%9A%E4%B8%AA%E6%96%B0%E9%97%BB%E6%9D%A5%E6%BA%90%E6%94%B6%E9%9B%86100%E5%A4%9A%E4%B8%87%E7%AF%87%E6%96%B0%E9%97%BB%E6%96%87%E7%AB%A0%E7%9A%84%E5%A4%A7%E5%9E%8B%E7%BD%91%E7%AB%99%E3%80%82Zhang%E3%80%81Zhao%E5%92%8CLeCun%E5%9C%A8%E8%AE%BA%E6%96%87%E4%B8%AD%E6%9E%84%E5%BB%BA%E4%BA%86%E4%B8%80%E4%B8%AA%E8%BE%83%E5%B0%8F%E7%9A%84%E9%9B%86%E5%90%88%EF%BC%8C%E5%85%B6%E4%B8%AD%E9%87%87%E6%A0%B7%E4%BA%86%E2%80%9C%E4%B8%96%E7%95%8C%E2%80%9D%E3%80%81%E2%80%9C%E4%BD%93%E8%82%B2%E2%80%9D%E3%80%81%E2%80%9C%E5%95%86%E4%B8%9A%E2%80%9D%E5%92%8C%E2%80%9C%E7%A7%91%E5%AD%A6%E2%80%9D%E7%AD%89%E6%96%B0%E9%97%BB%E6%96%87%E7%AB%A0%EF%BC%8C%E5%B9%B6%E5%B0%86%E5%85%B6%E7%94%A8%E4%BD%9C%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E5%9F%BA%E5%87%86%E3%80%82%E8%BF%99%E4%B8%AA%E6%95%B0%E6%8D%AE%E9%9B%86%E2%80%9Cag_news%E2%80%9D%E5%B7%B2%E7%BB%8F%E6%88%90%E4%B8%BA%E4%B8%80%E4%B8%AA%E7%BB%8F%E5%B8%B8%E4%BD%BF%E7%94%A8%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86%EF%BC%8C%E5%8F%AF%E4%BB%A5%E5%9C%A8Kaggle%E3%80%81PyTorch%E3%80%81Huggingface%E5%92%8CTensorflow%E4%B8%AD%E4%BD%BF%E7%94%A8%E3%80%82%E8%AE%A9%E6%88%91%E4%BB%AC%E4%B8%8B%E8%BD%BD[%E6%95%B0%E6%8D%AE%E4%BB%8EKaggle]/(https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset/)%E3%80%82%E8%AE%AD%E7%BB%83%E6%A0%B7%E6%9C%AC%E5%92%8C%E6%B5%8B%E8%AF%95%E6%A0%B7%E6%9C%AC%E5%88%86%E5%88%AB%E6%9C%8912%E4%B8%87%E7%AF%87%E5%92%8C7600%E7%AF%87%E6%96%B0%E9%97%BB%E6%96%87%E7%AB%A0%E3%80%82)

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br><span>3</span><br><span>4</span><br><span>5</span><br><span>6</span><br><span>7</span><br><span>8</span><br><span>9</span><br></pre></td><td><pre><span><span>import</span> pandas <span>as</span> pd</span><br><span><span>import</span> numpy <span>as</span> np</span><br><span><span>import</span> faiss</span><br><span>pd.set_option(<span>'display.max_colwidth'</span>, -<span>1</span>)</span><br><span>path = <span>"/content/gdrive/My Drive/data"</span></span><br><span>train = pd.read_csv(path + <span>"/gensim/ag_news_train.csv"</span>)</span><br><span><span>print</span>(train.shape)</span><br><span><span>print</span>(train.columns)</span><br><span>train[<span>'Description'</span>][<span>0</span>:<span>5</span>]</span><br></pre></td></tr></tbody></table>

输出形状为(120000,3)，列为\[‘ Class Index ‘， ‘ Title ‘， ‘ Description ‘\]。我们对“描述”栏感兴趣。以下是排名前五的记录。

*   _路透社——卖空者，华尔街日益减少的\\band极端愤世嫉俗者，又看到了绿色_
*   _路透——私人投资公司凯雷投资集团(\\which)以在国防工业投资时机恰当、偶尔引发争议而闻名，该公司已悄然将赌注押在了市场的另一个领域_
*   _路透社——油价飙升，加上对\\about经济和盈利前景的担忧，预计将在下周\\summer经济低迷的深度\\hang拖累股市_
*   _路透社——一位石油官员周六表示，在\\intelligence显示反叛民兵可能袭击\\infrastructure后，当局已经停止了伊拉克南部主要管道\\flows的石油出口_
*   _法新社——在距离美国总统大选仅剩三个月的时间里，世界油价不断刷新纪录，人们的钱包越来越紧，这给经济带来了新的威胁_

**数据嵌入**

出于说明的目的，我只使用10,000条记录进行Embeddings。

<table><tbody><tr><td><pre><span>1</span><br></pre></td><td><pre><span>sentences = train['Description'][0:10000]</span><br></pre></td></tr></tbody></table>

您需要pip安装“sentence\_transformers”库。

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br></pre></td><td><pre><span>!pip install sentence_transformers</span><br><span>from sentence_transformers import SentenceTransformer</span><br></pre></td></tr></tbody></table>

然后让我们使用预训练模型“bert-base-nli-mean-tokens”来声明模型。在[本页](https://www.sbert.net/docs/pretrained_models.html)上有许多预先训练好的模型。

<table><tbody><tr><td><pre><span>1</span><br></pre></td><td><pre><span>model = SentenceTransformer('bert-base-nli-mean-tokens')</span><br></pre></td></tr></tbody></table>

然后我们将“句子”编码为“sentence\_embeddings”。

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br><span>3</span><br></pre></td><td><pre><span>sentence_embeddings = model.encode(sentences)</span><br><span><span>print</span>(sentence_embeddings.shape)</span><br><span>sentence_embeddings[<span>0</span>:<span>5</span>]</span><br></pre></td></tr></tbody></table>

输出是10,000个列表。每个列表或向量的维数为768。下面是前5个Embeddings的输出。

_array(\[\[-0.26105028, 0.8585296 , 0.03941074, …, 1.0689917 , 1.1770816 , -0.74388623\], \[-0.2222097 , -0.03594436, 0.5209106 , …, 0.15727971, -0.3867779 , 0.49948674\], \[-0.3001758 , -0.41582862, 0.86036515, …, -0.6246218 , 0.52692914, -0.36817163\], \[ 0.3295024 , 0.22334357, 0.30229023, …, -0.41823167, 0.01728885, -0.05920589\], \[-0.22277102, 0.7840586 , 0.2004052 , …, -0.9121561 , 0.2918987 , -0.12284964\]\], dtype=float32)_

这有助于保存Embeddings以备将来使用。

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br></pre></td><td><pre><span><span>with</span> <span>open</span>(path + <span>'/AG_news.npy'</span>, <span>'wb'</span>) <span>as</span> file:</span><br><span>    np.save(file, sentence_embeddings)</span><br></pre></td></tr></tbody></table>

在上面的代码中，我使用了“npy”文件扩展名，这是NumPy数组文件的常规扩展名。下面是加载数据的代码:

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br></pre></td><td><pre><span><span>with</span> <span>open</span> (path + <span>'/AG_news.npy'</span>, <span>'rb'</span>) <span>as</span> f:</span><br><span>    sentence_embeddings = np.load(f, allow_pickle=<span>True</span>)</span><br></pre></td></tr></tbody></table>

有了这些Embeddings，我们就可以在HNSW数据结构中组织它们了。

**使用FAISS构建NHSW数据结构索引**

您需要像下面这样pip安装FAISS库:

<table><tbody><tr><td><pre><span>1</span><br></pre></td><td><pre><span>!pip install faiss-cpu --no-cache</span><br></pre></td></tr></tbody></table>

我将使用HNSWFlat(dim, m)类来构建HNSW。它需要预先确定的参数dim表示向量的维数，m表示数据元素与其他元素连接的边数。

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br><span>3</span><br><span>4</span><br></pre></td><td><pre><span><span>import</span> faiss</span><br><span>m = <span>32</span></span><br><span>dim = <span>768</span></span><br><span>index = faiss.IndexHNSWFlat(dim, m)</span><br></pre></td></tr></tbody></table>

如前所述，HNSW指数的创建分为两个不同的阶段。在初始阶段，该算法采用概率分布来预测引入新数据节点的最上层。在接下来的阶段，收集每个数据点的最近邻居，然后用一个表示为m的值进行修剪(在我们的例子中是m=16)。整个过程是迭代的，从插入层开始，一直到底层。

HNSW中有两个重要参数“efConstruction”和“efSearch”。这两个参数控制着索引结构构建的效率和有效性。它们帮助您在HNSW索引结构中的索引构建和最近邻搜索操作的速度和质量之间取得平衡。

1.  efConstruction:该参数用于HNSW索引的构建。它控制了构建索引结构的速度和结构质量之间的权衡。” efConstruction “值决定了在构建阶段要考虑多少候选项目。较高的“efConstruction”值将产生更准确的索引结构，但也会使构建过程变慢。
2.  efSearch:该参数用于在HNSW索引中查找查询点的最近邻居。“efSearch”值控制搜索速度和搜索质量之间的权衡。较高的“efSearch”值将导致更准确和详尽的搜索，但也会更慢。我们将“efConstruction”和“efSearch”分别设置为40和16:

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br></pre></td><td><pre><span>index.hnsw.efConstruction = <span>40</span> </span><br><span>index.hnsw.efSearch = <span>16</span>  </span><br></pre></td></tr></tbody></table>

我们已经声明了上面的数据结构。现在我们准备将数据“sentence\_embeddings”一个接一个地插入到数据结构中:

<table><tbody><tr><td><pre><span>1</span><br></pre></td><td><pre><span>index.add(sentence_embeddings)</span><br></pre></td></tr></tbody></table>

一旦完成，我们可以检查HNSW数据结构中有多少数据元素:

<table><tbody><tr><td><pre><span>1</span><br></pre></td><td><pre><span>index.ntotal</span><br></pre></td></tr></tbody></table>

输出为10000。它是“sentence\_embeddings”中的数据点数。接下来，HNSW建造了多少层?让我们来检查最大级别:

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br></pre></td><td><pre><span></span><br><span>index.hnsw.max_level</span><br></pre></td></tr></tbody></table>

最高级别为2.0。这意味着有第0层，第1层和第2层。接下来，您可能想知道每层中数据元素的数量。让我们来看看:

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br></pre></td><td><pre><span>levels = faiss.vector_to_array(index.hnsw.levels)</span><br><span>np.bincount(levels)</span><br></pre></td></tr></tbody></table>

输出为array(\[0,9713,280,7\])。“0”没有意义，你可以忽略它。它说第0层有9713个数据元素，第1层有280个元素，第2层只有7个元素。注意，9713 + 280 + 7 = 10000。您是否发现，较高层的数据元素数量比前几层呈指数级减少?这是因为数据元素的层分配采用指数衰减概率分布。

**FAISS为HNSW搜索示例**

假设我们的搜索查询是“经济繁荣与股市（economic booming and stock market）”。我们希望找到与我们的搜索查询相关的文章。我们将首先嵌入搜索查询:

<table><tbody><tr><td><pre><span>1</span><br></pre></td><td><pre><span>qry1 = model.encode(["economic booming and stock market"])</span><br></pre></td></tr></tbody></table>

使用代码index.search()，搜索过程非常简单。这里k是最近邻居的个数。我们将其设置为5以返回5个邻居。index.search()函数返回两个值” d “和” I “。

*   “d”:查询向量与k个最近邻居之间的距离列表。默认的距离度量是欧几里得距离。
*   “I”:它是索引中k个最近邻居的位置对应的索引列表。这些索引可用于查找数据集中的实际数据点。

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br><span>3</span><br><span>4</span><br><span>5</span><br></pre></td><td><pre><span>%%time</span><br><span>k=5</span><br><span>d, I = index.search(qry1, k)</span><br><span>print(I)</span><br><span>print(d)</span><br></pre></td></tr></tbody></table>

索引列表的输出是\[\[1467 4838 4464 7461 8299\]\]。我们将使用这些索引打印出搜索结果。

注意，我使用“%%time”来度量执行时间。它输出

\*CPU时间:user: 5.57 ms, sys: 5µs, total: 5.58 ms

这意味着搜索只需要几毫秒。这确实是令人难以置信的快!

距离输出列表为:\[\[158.19066 163.69077 164.47517 164.64172 164.64172\]\]

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br></pre></td><td><pre><span>for i in I[0]:</span><br><span>  print(train['Description'][i])</span><br></pre></td></tr></tbody></table>

输出：

_‘Rising oil prices are expected to hit China’s growth rate this year.’_

_‘Developing countries are starting to flex their financial muscles and invest overseas._

_‘The Tehran Stock Exchange has performed magnificently, but the market’s list of risks is outsized.’_

_‘Federal Express raised its earnings forecast, citing strong demand for its international express, ground and less-than-truckload services.’_

_‘Federal Express raised its earnings forecast, citing strong demand for its international express, ground and less-than-truckload services.’_ (Our data have duplications)

这些文章都是关于经济和股票市场的新闻。搜索速度以毫秒计非常快。这不仅仅是结果在哪里的问题，而是如何快速得到结果的问题，不是吗?

您可以通过[此链接](https://github.com/dataman-git/codes_for_articles/blob/master/HNSW.ipynb)下载笔记本进行上述搜索。

**总结**

我希望这篇文章能帮助你理解近似近邻(ANN)，以及它是如何提供高效搜索的。这篇文章解释了不同的人工神经网络算法，包括基于图的HNSW，基于哈希的LSH或产品量化，以及基于树的KD-Trees。这篇文章解释了HNSW如何构建其数据结构并逐个插入数据元素。本文演示了如何使用FAISS库构建用于查询搜索的HNSW。在下一篇文章“[搜索像光速- (2)LSH，](https://dataman-ai.medium.com/search-like-light-speed-2-lsh-b66c90349c66?sk=06225de6acda20982f04699b20428dc4)”中，我将讨论基于哈希的算法。

**附录**

在Malkov和Yashunin\[3\]的论文中，算法1到5伪代码中提供了HNSW方法。伪代码给出了算法的具体定义。我将这些描述添加到伪代码中，因为一些读者可能会发现它们有助于理解HNSW。算法1、算法2和算法3或算法4中的一个用于完成数据结构。一旦数据结构完成，以后的任何查询搜索都只使用算法5。

*   算法1:“INSERT”函数构建数据结构
*   算法2:“SEARCH-LAYER”函数计算KNN并存储邻居
*   算法3:“SEARCH-NEIGHBORS-SIMPLE”是一种选择邻居的简单方法
*   算法4:“SELECT-NEIGHBORS-HEURISTIC”函数是一种更复杂的选择邻居的方法
*   算法5:“KNN-SEARCH”函数进行查询搜索

让我们从算法1开始。

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br><span>3</span><br><span>4</span><br><span>5</span><br><span>6</span><br><span>7</span><br><span>8</span><br><span>9</span><br><span>10</span><br><span>11</span><br><span>12</span><br><span>13</span><br><span>14</span><br><span>15</span><br><span>16</span><br><span>17</span><br><span>18</span><br><span>19</span><br><span>20</span><br><span>21</span><br><span>22</span><br><span>23</span><br><span>24</span><br><span>25</span><br><span>26</span><br><span>27</span><br><span>28</span><br><span>29</span><br></pre></td><td><pre><span>Algorithm 1: INSERT()</span><br><span></span><br><span>INSERT(hnsw, q, M, Mmax, efConstruction, mL)</span><br><span>Input: multilayer graph hnsw, new element q, number of established</span><br><span>connections M, maximum number of connections for each element</span><br><span>per layer Mmax, size of the dynamic candidate list efConstruction, nor-</span><br><span>malization factor for level generation mL</span><br><span>Output: update hnsw inserting element q</span><br><span>1 W ← ∅ // list for the currently found nearest elements</span><br><span>2 ep ← get enter point for hnsw</span><br><span>3 L ← level of ep // top layer for hnsw</span><br><span>4 l ← ⌊-ln(unif(0..1))∙mL⌋ // new element’s level</span><br><span>5 for lc ← L … l+1</span><br><span>6 W ← SEARCH-LAYER(q, ep, ef=1, lc)</span><br><span>7 ep ← get the nearest element from W to q</span><br><span>8 for lc ← min(L, l) … 0</span><br><span>9 W ← SEARCH-LAYER(q, ep, efConstruction, lc)</span><br><span>10 neighbors ← SELECT-NEIGHBORS(q, W, M, lc) // alg. 3 or alg. 4</span><br><span>11 add bidirectionall connectionts from neighbors to q at layer lc</span><br><span>12 for each e ∈ neighbors // shrink connections if needed</span><br><span>13 eConn ← neighbourhood(e) at layer lc</span><br><span>14 if │eConn│ &gt; Mmax // shrink connections of e</span><br><span>// if lc = 0 then Mmax = Mmax0</span><br><span>15 eNewConn ← SELECT-NEIGHBORS(e, eConn, Mmax, lc)</span><br><span>// alg. 3 or alg. 4</span><br><span>16 set neighbourhood(e) at layer lc to eNewConn</span><br><span>17 ep ← W</span><br><span>18 if l &gt; L</span><br><span>19 set enter point for hnsw to q</span><br></pre></td></tr></tbody></table>

它在多层图中插入一个新元素q，保持每个元素每层最多有M个连接，并且每个元素的连接总数不超过Mmax的属性。该算法还保证连接元素之间的距离不大于某一最大距离，并且每层的连接数是均衡的。步骤如下:

1.  W←∅:初始化一个空列表W来存储当前找到的最近的元素。
2.  ep←get enter point for hnsw:获取多层图hnsw的进入点(即起始点)。
3.  L←ep的电平:获取进入点ep的电平。
4.  l←ln(unitif(0..1))∙mL⌋:为新元素q生成一个介于0和mL之间的随机级别，其中mL是级别生成的归一化因子。
5.  for lc←L…L +1:循环从L到L +1的层。
6.  W←SEARCH LAYER(q, ep, ef=1, lc):使用进入点ep和最大距离ef=1在lc层中搜索离q最近的元素。将找到的元素存储在列表W中。
7.  ep←取W到q最近的元素:取W到q最近的元素。
8.  for lc←min(L, L)…0:循环遍历从min(L, L)到0的层。
9.  W←SEARCH LAYER(q, ep, efConstruction, lc):使用进入点ep和最大距离efConstruction搜索层lc中离q最近的元素。将找到的元素存储在列表W中。
10.  neighbors←SELECT neighbors (q, W, M, lc):选择W到q最近的M个邻居，只考虑lc层的元素。
11.  在lc层添加邻居到q的双向连接:在lc层添加q与所选邻居之间的双向连接。
12.  对于每个e∈neighbors: //如果需要收缩连接  
    对于q的每个邻居e，检查e的连接数是否超过Mmax。如果是这样，使用SELECT neighbors (e, eConn, Mmax, lc)选择一组新的邻居来收缩e的连接，其中eConn是e在lc层的当前连接集。
13.  eNewConn←SELECT NEIGHBORS(e, eConn, Mmax, lc):为e选择一组新的邻居，只考虑lc层的元素，保证连接数不超过Mmax。
14.  `set neighborhood (e) at layer lc to eNewConn`:将层lc的e的连接集更新为新的set eNewConn。
15.  `ep <- W`:设置hnsw的进入点为q。
16.  `if 1 > L`:将hnsw的起始点设为q，因为新元素q现在是图的一部分。
17.  `return hnsw`:返回更新后的多层图hnsw。

让我们看看算法2。

它在HNSW数据结构上执行K近邻搜索，以查找特定层lc中与查询元素q最近的K个元素。然后，它根据查询元素q与候选元素C和e之间的距离更新候选元素C的集合和找到的最近邻居列表W。最后，如果W中的元素数量超过了动态候选列表ef的大小，则该函数删除从W到q最远的元素。

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br><span>3</span><br><span>4</span><br><span>5</span><br><span>6</span><br><span>7</span><br><span>8</span><br><span>9</span><br><span>10</span><br><span>11</span><br><span>12</span><br><span>13</span><br><span>14</span><br><span>15</span><br><span>16</span><br><span>17</span><br><span>18</span><br><span>19</span><br><span>20</span><br><span>21</span><br><span>22</span><br><span>23</span><br><span>24</span><br></pre></td><td><pre><span>Algorithm 2: SEARCH-LAYER()</span><br><span></span><br><span>SEARCH-LAYER(q, ep, ef, lc)</span><br><span>Input: query element q, enter points ep, number of nearest to q ele-</span><br><span>ments to return ef, layer number lc</span><br><span>Output: ef closest neighbors to q</span><br><span>1 v ← ep // set of visited elements</span><br><span>2 C ← ep // set of candidates</span><br><span>3 W ← ep // dynamic list of found nearest neighbors</span><br><span>4 while │C│ &gt; 0</span><br><span>5 c ← extract nearest element from C to q</span><br><span>6 f ← get furthest element from W to q</span><br><span>7 if distance(c, q) &gt; distance(f, q)</span><br><span>8 break // all elements in W are evaluated</span><br><span>9 for each e ∈ neighbourhood(c) at layer lc // update C and W</span><br><span>10 if e ∉ v</span><br><span>11 v ← v ⋃ e</span><br><span>12 f ← get furthest element from W to q</span><br><span>13 if distance(e, q) &lt; distance(f, q) or │W│ &lt; ef</span><br><span>14 C ← C ⋃ e</span><br><span>15 W ← W ⋃ e</span><br><span>16 if │W│ &gt; ef</span><br><span>17 remove furthest element from W to q</span><br><span>18 return W</span><br></pre></td></tr></tbody></table>

以下是上述代码的步骤:

1.  初始化变量v为当前的入口点ep。
2.  初始化集合C为当前候选集合。
3.  初始化一个空列表W来存储找到的最近邻。
4.  循环直到候选集合C中的所有元素都求值为止。
5.  从候选元素集合c中提取离查询元素q最近的元素c。
6.  获取从找到的最近邻W到查询元素q的列表中最远的元素f。
7.  如果c到q的距离大于f到q的距离:
8.  然后打破这个循环。
9.  对于lc层c邻域内的每个元素e:
10.  如果e不在访问元素v的集合中，则:
11.  将e添加到访问元素v的集合中。
12.  设f为从W到q的最远的元素。
13.  如果e和q之间的距离小于等于f和q之间的距离，或者W中的元素个数大于等于ef(动态候选列表的大小)，则:
14.  将候选集C更新为C∈e。
15.  将发现的最近邻居W的列表更新为W∈e。
16.  如果W中的元素个数大于等于ef，则:
17.  移除从W到q的最远的元素。
18.  返回找到的最近邻居W的列表。

算法3.

这是一个简单的最近邻选择算法，它接受一个基本元素q、一组候选元素C和一些邻居M作为输入。它返回候选元素C集合中离q最近的M个元素。

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br><span>3</span><br><span>4</span><br><span>5</span><br><span>6</span><br><span>7</span><br></pre></td><td><pre><span>Algorithm 3: SELECT-NEIGHBORS-SIMPLE()</span><br><span></span><br><span>SELECT-NEIGHBORS-SIMPLE(q, C, M)</span><br><span>Input: base element q, candidate elements C, number of neighbors to</span><br><span>return M</span><br><span>Output: M nearest elements to q</span><br><span>return M nearest elements from C to q</span><br></pre></td></tr></tbody></table>

步骤如下:

1.  初始化一个空集R来存储选中的邻居。
2.  初始化一个工作队列W来存储候选元素。
3.  如果设置了extendCandidates标志(即true)，则通过将C中每个元素的邻居添加到队列W来扩展候选列表。
4.  而W的大小大于0,R的大小小于M:
5.  从W到q中提取最近的元素e。
6.  如果e比R中的任何元素更接近q，把e加到R中。
7.  否则，将e添加到丢弃队列Wd中。
8.  如果设置了keepPrunedConnections标志(即true)，则从Wd添加一些丢弃的连接到R。
9.  返回R。

让我们看看算法4。

这是一个更复杂的最近邻选择算法，它接受一个基本元素q、一组候选元素C、若干个邻居M、一个层数lc和两个标志extendCandidates和keepPrunedConnections作为输入。它返回由启发式选择的M个元素。

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br><span>3</span><br><span>4</span><br><span>5</span><br><span>6</span><br><span>7</span><br><span>8</span><br><span>9</span><br><span>10</span><br><span>11</span><br><span>12</span><br><span>13</span><br><span>14</span><br><span>15</span><br><span>16</span><br><span>17</span><br><span>18</span><br><span>19</span><br><span>20</span><br><span>21</span><br><span>22</span><br><span>23</span><br><span>24</span><br><span>25</span><br><span>26</span><br><span>27</span><br><span>28</span><br></pre></td><td><pre><span>Algorithm 4: SELECT-NEIGHBORS-HEURISTIC()</span><br><span></span><br><span>SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates, keep-</span><br><span>PrunedConnections)</span><br><span>Input: base element q, candidate elements C, number of neighbors to</span><br><span>return M, layer number lc, flag indicating whether or not to extend</span><br><span>candidate list extendCandidates, flag indicating whether or not to add</span><br><span>discarded elements keepPrunedConnections</span><br><span>Output: M elements selected by the heuristic</span><br><span>1 R ← ∅</span><br><span>2 W ← C // working queue for the candidates</span><br><span>3 if extendCandidates // extend candidates by their neighbors</span><br><span>4 for each e ∈ C</span><br><span>5 for each eadj ∈ neighbourhood(e) at layer lc</span><br><span>6 if eadj ∉ W</span><br><span>7 W ← W ⋃ eadj</span><br><span>8 Wd ← ∅ // queue for the discarded candidates</span><br><span>9 while │W│ &gt; 0 and │R│&lt; M</span><br><span>10 e ← extract nearest element from W to q</span><br><span>11 if e is closer to q compared to any element from R</span><br><span>12 R ← R ⋃ e</span><br><span>13 else</span><br><span>14 Wd ← Wd ⋃ e</span><br><span>15 if keepPrunedConnections // add some of the discarded</span><br><span>// connections from Wd</span><br><span>16 while │Wd│&gt; 0 and │R│&lt; M</span><br><span>17 R ← R ⋃ extract nearest element from Wd to q</span><br><span>18 return R</span><br></pre></td></tr></tbody></table>

步骤如下:

1.  初始化三个队列:R用于选择的邻居，W用于工作的候选，Wd用于丢弃的候选。
2.  设置R的大小为0,W的大小为C的大小。
3.  如果extendCandidates被设置(即，true):
4.  对于C中的每个元素e:
5.  对于第lc层e的每一个邻居eadj:
6.  如果eadj不在W中，则在W中添加它。
7.  而W的大小大于0,R的大小小于M:
8.  从W到q中提取最近的元素e。
9.  如果e比R中的任何元素更接近q，把e加到R中。
10.  否则，将e加到Wd。
11.  如果设置了keepPrunedConnections(即true):
12.  而Wd的大小大于0,R的大小小于M:
13.  从Wd到q中提取最近的元素e。
14.  如果e比R中的任何元素更接近q，就把e加到R中。
15.  返回R。

最后，让我们看看算法5。

这个搜索算法与算法1基本相同。

<table><tbody><tr><td><pre><span>1</span><br><span>2</span><br><span>3</span><br><span>4</span><br><span>5</span><br><span>6</span><br><span>7</span><br><span>8</span><br><span>9</span><br><span>10</span><br><span>11</span><br><span>12</span><br><span>13</span><br><span>14</span><br></pre></td><td><pre><span>Algorithm 5: K-NN-SEARCH()</span><br><span></span><br><span>K-NN-SEARCH(hnsw, q, K, ef)</span><br><span>Input: multilayer graph hnsw, query element q, number of nearest</span><br><span>neighbors to return K, size of the dynamic candidate list ef</span><br><span>Output: K nearest elements to q</span><br><span>1 W ← ∅ // set for the current nearest elements</span><br><span>2 ep ← get enter point for hnsw</span><br><span>3 L ← level of ep // top layer for hnsw</span><br><span>4 for lc ← L … 1</span><br><span>5 W ← SEARCH-LAYER(q, ep, ef=1, lc)</span><br><span>6 ep ← get nearest element from W to q</span><br><span>7 W ← SEARCH-LAYER(q, ep, ef, lc =0)</span><br><span>8 return K nearest elements from W to q</span><br></pre></td></tr></tbody></table>

步骤如下:

1.  初始化一个空集合W(当前最近元素的集合)，并将进入点ep设置为网络的顶层。
2.  设置进入点ep的水平为顶层L。
3.  对于每一层lc，从L到1(即从顶层到底层):
4.  使用查询元素q和当前最近的元素W搜索当前层，并将最近的元素添加到W中。
5.  将进入点ep设置为W到q最近的元素。
6.  使用查询元素q和当前最近的元素W搜索下一层，并将最近的元素添加到W中。
7.  返回W中最接近的K个元素作为输出。

**引用**

*   \[1\] [Pugh, W. (1990). Skip lists: A probabilistic alternative to balanced trees. _Communications of the ACM, 33_(6), 668–676. doi:10.1145/78973.78977. S2CID 207691558.](https://15721.courses.cs.cmu.edu/spring2018/papers/08-oltpindexes1/pugh-skiplists-cacm1990.pdf)
*   \[2\] [Xiang Zhang, Junbo Zhao, & Yann LeCun. (2015). Character-level Convolutional Networks for Text Classification](https://www.tensorflow.org/datasets/catalog/ag_news_subset)
*   \[3\] [Yu. A. Malkov, & D. A. Yashunin. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.](https://arxiv.org/abs/1603.09320)
*   \[4\] Aristides Gionis, Piotr Indyk, Rajeev Motwani, et al. 1999. Similarity search in high dimensions via hashing. In Vldb, Vol. 99. 518–529.

* * *

# 大白话理解HNSW大白话唠唠HNSW的来龙去脉，不涉及算法证明，只说明算法工作方式和原理，为了分析一些HNSW的源码。 - 掘金

Title: 大白话理解HNSW大白话唠唠HNSW的来龙去脉，不涉及算法证明，只说明算法工作方式和原理，为了分析一些HNSW的源码。 - 掘金

URL Source: https://juejin.cn/post/7082579087776022541

Markdown Content:
大白话理解HNSW大白话唠唠HNSW的来龙去脉，不涉及算法证明，只说明算法工作方式和原理，为了分析一些HNSW的源码。 - 掘金
===============
   

 [![Image 1: 稀土掘金](https://lf-web-assets.juejin.cn/obj/juejin-web/xitu_juejin_web/e08da34488b114bd4c665ba2fa520a31.svg) ![Image 2: 稀土掘金](https://lf-web-assets.juejin.cn/obj/juejin-web/xitu_juejin_web/6c61ae65d1c41ae8221a670fa32d05aa.svg)](https://juejin.cn/) 

*   首页
    
    *   [首页](https://juejin.cn/)
    *   [BOT](https://juejin.cn/bots)
    *   [沸点](https://juejin.cn/pins)
    *   [课程](https://juejin.cn/course)
    *   [直播](https://juejin.cn/live)
    *   [活动](https://juejin.cn/events/all)
    *   [竞赛](https://juejin.cn/challenge)
    
    [商城](https://detail.youzan.com/show/goods/newest?kdt_id=104340304)
    
    [APP](https://juejin.cn/app?utm_source=jj_nav)
    
    [插件](https://juejin.cn/extension?utm_source=jj_nav)
    
    *   [![Image 3: image](http://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/eadba9c67542181d4f773b0d25e79503.png~tplv-k3u1fbpfcp-jj:115:0:0:0:q75.avis)](https://juejin.cn/post/7408619644196274176?utm_source=top&utm_medium=web&utm_campaign=marscode09)

*   *   搜索历史 清空
        
    *   创作者中心
        
        *   写文章
            
        *   发沸点
            
        *   写笔记
            
        *   写代码
            
        *   草稿箱
            
        
        创作灵感 查看更多
        
*   ![Image 4: vip](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ffd3e238ee7f46eab42bf88af17f5528~tplv-k3u1fbpfcp-image.image#?w=25&h=26&s=5968&e=svg&a=1&b=dacbbc)
    
    会员
    
*   登录
    
    注册
    
    首次登录 / 注册免费领取
    -------------
    
    登录 / 注册
    

   

 

  

大白话理解HNSW

[沧叔解码](https://juejin.cn/user/2559318800998141/posts)

2022-04-04 3,791 阅读11分钟

关注

背景介绍
----

在语义检索领域，一般都是把待检索的文档集合提前编码成指定维度的向量入库。检索的时候把query也编码成同样维度的向量，然后在库里面根据指定的距离度量方式寻找距离最近的向量，也就是最相关的文档。

那么如何在使用尽量少的资源，又快又准地检索相关文档呢？

暴力检索
----

最最容易想到的解决方式，把query编码的向量拿到向量库中一一计算距离，然后返回topK距离最近的向量代表的文档作为最相关的topK文档。

这种查找方式当然能够找到真正的topK最相关的文档，但是可以想象这种效率有多低。当然可以也做一些优化，比如每次检索的时候向量库中的向量是随机取出进行距离计算，增加截断参数，只计算前1w个向量，然后从这1w个里面选择topK，但是这种优化方式并不是无损的，截断参数设置较低，影响召回的质量，截断参数设置较大，效率依然低下。

那还能怎么优化呢？

基于图的查找
------

向量其实就是多维空间中的一个点，向量的近邻检索就是寻找空间中相近的点。

我们来看个直观的例子，如下图所示，在二维平面（二维向量空间）中的这些黑色节点，上面的问题就是怎么从这些黑色的点中寻找离红点最近的点？

![Image 5: 无线点图.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c79c141be65249a6b1eec61826bd1369~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

一种简单做法是把黑色中的某些点和点连起来，构建一个查找图，存下来。与点直接相连的点叫做这个点的邻居节点。

查找的时候随机从某个黑色的点（起始遍历节点：entry point）出发，计算这个点和目标点的距离，然后再计算这个点的邻居节点和目标点的距离，选择距离目标节点距离最近的节点继续迭代。如果当前处理的黑色节点比所有邻居节点距离目标节点都近，则当前黑色节点就是距离最近的节点。

我们举个例子。假设我们对上面的这些黑色节点构建了如下的图。entry point节点是I，要寻找距离红色节点最近的节点。

![Image 6: 有线点图.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7d768e080f144fa2ba409620cbe253c1~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

查找算法执行如下：

先计算I和红色节点的距离，记录下来，再计算下I的邻居节点\[B, C\]和红色节点的距离，对比三个距离哪个最近，发现B最近，把B点距离红色节点的距离保存下来，继续处理B的邻居节点\[A, C, I, H, E\]和红色节点的距离，发现E点最近，则把E点和红色节点的距离保存下来，在继续计算E点的邻居节点\[J, B, G, D\]和红色节点的距离，此时我们发现E点的距离是最近的，则返回最近的点E。

从上面的例子我们可以发现，这种思路虽然行得通，但是存在一些问题：

1.  找到的结果不是最优的结果，最优的应该是L。
2.  如果是要返回最近的两个节点，而L和E之间没有连线，这将增加迭代次数，并且大大影响效率
3.  K点是个孤岛，如果随机初始的节点不是K点则它永远无法访问到，而K作为初始节点，则无法遍历其他节点，只能返回K，误差较大。
4.  如何确定哪些节点应该互为邻居呢？

对于上面的这些 问题，粗暴并且直观的解决方案可以是：

*   距离近到一定程度的节点必须互为邻居节点（解决问题2,4，降低1出现的概率）
*   所有的节点必须都有邻居节点（解决问题3）

上面直观的解决方案是否有更严谨，可实现的描述？

NSW（Navigable Small World graphs）
---------------------------------

### 德劳内算法

在图论中有一个剖分法可以有效解决上一节提到的那些问题，**德劳内（Delaunay）三角剖分算法**。这个算法对一批空间中的节点处理之后可以达到以下效果：

*   图中的每个节点都有邻居节点
*   距离相近的节点都互为邻居节点
*   图中的所有连接线段数量最少（邻居对最少）

实际的效果图如下：

![Image 7: 德劳内构建.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/86e79e68581d47cdab331cf10792001e~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

但是德劳内三角剖分法有两个缺点：

1.  图的构建时间复杂度太高
2.  查找效率比较低：如果起始点和目标点之间的距离比较远，需要大量的迭代才能找到目标。

由于以上的两个缺点，在NSW中并没有直接使用德劳内三角剖分法。为了解决德劳内存在的问题，NSW做了两个改进：

1.  NSW使用的是局部的信息来构建图，降低了构建的复杂度。（后面会介绍构建的步骤）
2.  使用局部信息构建图会产生一些“高速公路”，如下图中红色的线所示。使得相邻较远的节点也有互为邻居的概率，从而提升迭代效率。比如从下图的entry point开始查找距离绿色节点，通过红色箭头的线路可以很快找到。

![Image 8: image-20220401095021312.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/fd7a0726543249a39df623b617203a91~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

### NSW构建算法描述

NSW的构建算法非常简单，遍历所有待插入的节点，当新增一个节点时，从当前图中任意节点出发，寻找距离要新增节点的最近m个节点作为邻居节点，把新节点加入图中，并连接新节点和所有的邻居节点。

### NSW构建例子

文字描述有点苍白，我们看个例子，节点的处理顺序为字母顺序，我们规定最多只查询3个邻居：

*   **黑色节点**表示待插入的节点
*   **红色节点**表示当前处理的节点
*   **绿色节点**和**实线**表示已经构建好的图
*   **虚线**表示当前处理节点和邻居节点的连线
*   **红色连线**表示“高速公路”

![Image 9: nsw构建1 (1).png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d13c45daf81c44989f303161dc01c341~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

![Image 10: nsw构建1 (2).png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6c2d8098449d444495226e03cc7f82a6~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

![Image 11: nsw构建1 (4).png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/cd068a55519c4833bd744a7ba10287f3~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

![Image 12: nsw构建1 (7).png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/66958c93ee644376a592c33d3406bb3f~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

![Image 13: nsw构建1 (8).png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4f17538f0c2e4bbf9dd844a6f68e06e9~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

![Image 14: nsw构建1 (9).png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/629bd74390ba43dfb08fa362e93be367~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

![Image 15: nsw构建1 (10).png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2bf6b85805b441d4910d2bfc81238046~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

![Image 16: nsw构建1 (11).png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f03e71f829614ef489e488646e778651~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

### 德劳内和NSW构建结果对比

我们对比下德劳内构建的结果：

![Image 17: 德劳内构建.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b3ce4c44717b4a0aacb4e9c91956d13a~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

如果从I开始查找F附近的节点，则德劳内构建的图需要多次迭代，但是NSW通过高速公路可以快速找到。

### 大白话理解NSW

我们从感性上重新理解下NSW的算法，NSW构建的过程中，节点加入是随机的，为当前加入的节点寻找邻居都是使用局部信息，所以前期加入的节点寻找到的邻居有比较高的概率不是最近的，可能是全局来看比较远的节点互为邻居，这就是“高速公路”。而且新增节点都是无脑查询最近的邻居，算法复杂度比较低。

### NSW的查找算法

NSW的查找算法主要是在构建好的NSW图中查找指定的目标节点q的k个近邻点。

NSW的查找算法依赖两个堆和一个位图，用来优化查询速度：

*   visited：已经查找过的节点位图
*   candidates：候选节点的最小堆，堆顶节点是距离目标节点最近的候选节点
*   results：当前结果节点的最大堆，堆顶节点是距离目标节点最远的节点

算法步骤：

1.  建立最大堆results，建立最小堆candidates，建立位图visited
2.  随机选择一个节点作为查找的起点，加入visited，并计算到目标节点的距离，加入candidates
3.  从candidates中获取堆顶的候选节点（candidates中距离目标节点最近的节点）
4.  如果当前候选节点到目标距离大于results的堆顶节点（results中距离目标节点最远的节点）到目标节点的距离并且results的大小已经满足topK，则结束迭代。
5.  否则，遍历候选节点的所有邻居，如果邻居节点没有在visited中，则把邻居节点加入candidates和visited中。
6.  返回3

在论文中会对3-5部加个迭代限制的参数m，也就是最多执行m轮查找，这样可以兼顾到查找的时延和准确度的权衡。另外，整个查询步骤和论文中的伪代码描述有点差别，有兴趣的可以去参考资料中看下论文的原文描述。

NSW已经是一个比较优秀的近邻查找算法了，但是大佬们肯定不会满足于此，于是就有了HNSW。

HNSW（Hierarchcal Navigable Small World graphs）
----------------------------------------------

在介绍HNSW和NSW的关系之前，我们可以先对比下有序链表和跳表的关系。

### 有序链表和跳表

我们看下对于同一批数据，有序链表和跳表的结构和查询效率。

![Image 18: 跳表.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c4b99896cecb496caa6b2767da7988ea~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

如上图所示，跳表是由多个有序链表组成的，最底层的有序链表包含了所有的节点。跳表的构建有很多种优化版本，但是最朴素的构建方法非常简单粗暴。从最底层开始遍历每个节点，对每个节点抛硬币，如果正面则进入上一层的有序链表，如此处理每一层的数据。

对于要查找的目标，有序链表只能通过header，从头遍历，直到找到目标或者找到大于目标的最小节点（找不到的情况），然后停止查找。而对于跳表来说它从上往下找，在每一层中找到第一个小于等于目标节点的节点，然后往后或者往下继续查找，停止条件和有序链表一样。

文字描述还是比较枯燥，我们看图中的例子，比如我们的目标节点是59。有序链表就是从头往后找，需要查找7次。而跳表从上往下找，需要5次就找到了目标。可能在这个例子中跳表的效率不明显，但是可以想象成千上百万的节点场景跳表的效率。所以，跳表通过增加层数，以空间换时间，提高了查找的效率。

跳表的上层有序链表之间的节点跨度比较大，就像NSW中的“高速通道”，同样提高了查找的效率。HNSW就是在NSW的基础上提出分层的概念，进一步提高了查找的效率。

### HNSW结构

![Image 19: hnsw论文图.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/15850321d1714cd3b98a109d0be96fae~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

上面这个图是HNSW论文中的图，第0层是包含了所有的节点。在第i层中的节点，在所有j <= i 层中都存在。每一层可以理解成都是一个NSW（构建算法有所不同，后面会介绍）。

### HNSW查找算法

比如我们要从红色节点开始查找绿色节点。则在最顶层查找红色节点的最近节点，然后以该最近节点进入下一层继续寻找最近的节点，直到最底层中找到的最近节点就是目标。

HNSW中每一层的查找算法和NSW的一模一样。

### HNSW构建算法

HNSW的构建依赖一个随机函数，这个函数产生一个随机值，表示当前处理节点可以到达的层数。然后在每一层为当前处理节点寻找邻居节点，做连线，每一层的构建算法和NSW整体步骤是一样的，差别在于HNSW对于邻居的选取除了直接寻找最近邻外，提出了另一种启发式的选择算法。

#### 启发式邻居选择算法

一句话描述就是对于节点q启发式寻找邻居，找的是邻居候选列表中c，满足c到q的距离比c到当前已经确定的邻居集合中的左右邻居都要近。

乍一看可能有点绕，看个示例图：

![Image 20: 启发式邻居选择算法.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4b3807a214da4502a19a0cd8e494eabf~tplv-k3u1fbpfcp-jj-mark:3024:0:0:0:q75.awebp)

如上图所示，要为红色节点Q从邻居候选列表（A，B，C，D，E，F，G）中选择四个邻居。当前已经确定的邻居有A和B。

最近邻选择算法会继续选择C和F，而启发式选择算法会继续选择D和E。

从结果来看，最近邻的邻居选择算法选择的邻居集合是比较聚集的，而启发式选择算法选择的邻居是比较发散的，因此启发式算法可以快速查找位于各个方向的目标节点。比如目标节点位于E点附近，最近邻的邻居选择算法就会通过更多的迭代才能找到，而启发式算法可以快速定位。

最后
--

如有疏漏，欢迎指正讨论。

参考资料
----

1.  [NSW 论文](https://link.juejin.cn/?target=https%3A%2F%2Fpublications.hse.ru%2Fmirror%2Fpubs%2Fshare%2Ffolder%2Fx5p6h7thif%2Fdirect%2F128296059 "https://publications.hse.ru/mirror/pubs/share/folder/x5p6h7thif/direct/128296059")
2.  [HNSW 论文](https://link.juejin.cn/?target=https%3A%2F%2Farxiv.org%2Fftp%2Farxiv%2Fpapers%2F1603%2F1603.09320.pdf "https://arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf")

标签：

[算法](https://juejin.cn/tag/%E7%AE%97%E6%B3%95)

