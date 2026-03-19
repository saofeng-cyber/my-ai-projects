from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import get_ali_clients

ali_model, ali_embeddings = get_ali_clients()

loader = TextLoader("../data/deepseek百度百科.txt", encoding="utf-8", )
docs = loader.load()
# 分割器
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
documents = splitter.split_documents(docs)
# BM25Retriever.k = 3
doc_BM25Retriever = BM25Retriever.from_documents(documents=documents, k=3)
vector_store = Chroma(collection_name="multiQuery", embedding_function=ali_embeddings)
vector_store.add_documents(documents=documents)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retriever = EnsembleRetriever(retrievers=[doc_BM25Retriever, retriever], weights=[0.5, 0.5])
prompt = ChatPromptTemplate.from_template("请根据下面的文档内容和问题进行回答: \n\n文档内容{doc}, \n\n问题{question}")
chain = RunnableParallel({"question": lambda x: x["question"], "doc": lambda x: retriever.invoke(x["question"])}) | prompt | ali_model
res = chain.invoke({"question": "deepseek的应用场景"})
print(res)
