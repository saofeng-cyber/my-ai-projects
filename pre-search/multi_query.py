from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.load import dumps, loads

from models import get_ali_clients

# 获取模型和模型向量
ali_model, ali_embedding = get_ali_clients()

loader = TextLoader("../data/deepseek百度百科.txt", encoding="utf-8", )
docs = loader.load()

# 分割器
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
documents = splitter.split_documents(docs)

vector_store = Chroma(collection_name="multiQuery", embedding_function=ali_embedding)
vector_store.add_documents(documents=documents)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
template = """
你是一个AI语言模型助手，你的任务是根据用户问题生成3个不同版本的问题，
便于从向量中检索出用户问题的答案的原始文档数据库，通过对用户问题产生多个观点，
你的目标是提供帮助用户克服了基于距离的相似性检索的一些限制。 原始问题：{question}, 提供一个3个问题的序列
"""
prompt = PromptTemplate(template=template, input_variables=["question"])
@chain
def get_questions(content):
    return content.split("\n")


generate_queries = prompt | ali_model | StrOutputParser() | get_questions
# questions = chain.invoke(input={"question": "deepseek的应用场景"})
print(f"{'*' * 100}")

@chain
def get_unique_answer(content:list[list[Document]]):
    flattened_docs = [dumps(doc) for sunList in content for doc in sunList]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]


chain = generate_queries | retriever.map() | get_unique_answer
context_docs = chain.invoke({"question": "deepseek的应用场景"})

prompt1 = ChatPromptTemplate.from_template("请根据下面的文档内容和问题进行回答: \n\n文档内容{doc}, \n\n问题{question}")
chain1 = prompt1 | ali_model
res = chain1.invoke({"doc": context_docs, "question": "deepseek的应用场景"})
print(res.content)