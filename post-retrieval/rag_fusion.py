from langchain_chroma import Chroma
from langchain_classic import hub
from langchain_core.runnables import chain

from models import get_ali_clients

ali_model, ali_embeddings = get_ali_clients()

texts = [
    "人工智能在医疗诊断中的应用。",
    "人工智能如何提升供应链效率。",
    "NBA季后赛最新赛况分析。",
    "传统法式烘焙的五大技巧。",
    "红楼梦人物关系图谱分析。",
    "人工智能在金融风险管理中的应用。",
    "人工智能如何影响未来就业市场。",
    "人工智能在制造业的应用。",
    "今天天气怎么样",
    "人工智能伦理：公平性与透明度。"
]

# 创建向量数据库对象
vectorstore = Chroma.from_texts(texts=texts, embedding=ali_embeddings)
retriever = vectorstore.as_retriever()
#从langchain官网拉取预先定义好的prompt
prompt = hub.pull("langchain-ai/rag-fusion-query-generation")
print(prompt)
help(retriever)