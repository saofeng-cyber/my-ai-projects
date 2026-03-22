import os

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document

# 设置 API Key
os.environ["OPENAI_API_KEY"] = "sk-..."

# 1. 准备原始文档
raw_text = """
[第一章] 人工智能的历史与发展
人工智能的概念最早可以追溯到 1956 年的达特茅斯会议。
早期的人工智能主要基于规则系统，专家系统在 80 年代非常流行。
然而，由于计算能力的限制和数据缺乏，AI 经历了第一次寒冬。
[第二章] 机器学习的崛起
进入 21 世纪，随着互联网大数据的爆发和 GPU 算力的提升，机器学习开始主导 AI 领域。
深度学习（Deep Learning）在图像识别和自然语言处理上取得了突破性进展。
2017 年 Transformer 架构的提出，彻底改变了 NLP 的格局，为大语言模型奠定了基础。
[第三章] 生成式 AI 的未来
当前，生成式 AI 正在重塑各行各业。从代码生成到艺术创作，应用无处不在。
未来的挑战在于对齐（Alignment）、安全性以及如何降低推理成本。
"""
docs = [Document(page_content=raw_text)]

# 2. 初始化存储组件
# 向量存储：存放子块（Child）的向量
vectorstore = Chroma(
    collection_name="parent_child_collection",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_parent_child"
)

# 文档存储：存放父块（Parent）的完整文本 (Key-Value Store)
# 这里使用内存存储，生产环境可用 Redis, Cassandra 等
store = InMemoryStore()

# 3. 定义分块策略
# 父块分块器：大块，用于最终生成上下文 (例如 1000 字符)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 子块分块器：小块，用于检索匹配 (例如 200 字符)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

# 4. 构建父子检索器
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 5. 执行索引 (Add Documents)
# 内部逻辑：
# 1. 用 parent_splitter 切分文档 -> 得到父块 -> 存入 store (key=uuid, value=parent_text)
# 2. 对每个父块，用 child_splitter 切分 -> 得到子块
# 3. 对子块向量化 -> 存入 vectorstore (metadata 中包含父块的 uuid)
retriever.add_documents(docs)

print(f"索引完成。向量库中子块数量: {vectorstore._collection.count()}")
print(f"文档存储中父块数量: {len(list(store.yield_keys()))}")

# 6. 执行检索
query = "Transformer 架构是在什么背景下提出的？它有什么影响？"
print(f"\n用户查询: {query}")

# 内部逻辑：
# 1. 向量库检索 Top-K 子块
# 2. 提取子块中的 parent_uuid
# 3. 从 store 中取出对应的完整父块文本
retrieved_docs = retriever.invoke(query)

print("\n--- 检索到的上下文 (父块原文) ---")
for i, doc in enumerate(retrieved_docs):
    print(f"[父块 {i+1}]:\n{doc.page_content}\n{'-'*30}")

# 此时 retrieved_docs 包含的是大段的完整原文，可以直接喂给 LLM