from langchain_classic.retrievers.multi_vector import SearchType
from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.stores import InMemoryStore

from models import get_ollama_completion, get_ali_embeddings

loader = TextLoader("../data/deepseek百度百科.txt", encoding="utf-8")
docs = loader.load()

# 获取大模型
ollama_llm = get_ollama_completion()
ali_embedding = get_ali_embeddings()

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)
vectorstore = Chroma(collection_name="parent_child", embedding_function=ali_embedding)
store = InMemoryStore()
retriever = ParentDocumentRetriever(vectorstore=vectorstore,
                                    docstore=store,
                                    child_splitter=child_splitter,
                                    parent_splitter=parent_splitter, id_key="doc_id", search_kwargs={"k": 5},
                                    search_type=SearchType.similarity)

retriever.add_documents(documents=docs)

prompt = ChatPromptTemplate.from_template("请根据下面的文档内容和问题进行回答: \n\n文档内容{doc}, \n\n问题{question}")
chain = RunnableParallel(
    {"doc": lambda x: retriever.invoke(x["question"]), "question": lambda x: x["question"]}) | prompt | ollama_llm

stream = chain.stream(({"question": "deepseek的主要产品，列举一下"}))
# 存储思考过程和最终答案
reasoning_content = ""
answer_content = ""
# 遍历流式响应
for chunk in stream:
    # 检查是否有思考内容
    if hasattr(chunk, 'additional_kwargs') and 'reasoning_content' in chunk.additional_kwargs:
        reasoning_part = chunk.additional_kwargs['reasoning_content']
        if reasoning_part:  # 如果有思考内容
            reasoning_content += reasoning_part
            print(f"{reasoning_part}", end="", flush=True)

    # 检查是否有实际回答内容
    if chunk.content:  # chunk.content 包含实际回答
        answer_content += chunk.content
        print(f"\r答案: {answer_content}", end="", flush=True)

print(f"\n\n{'*' * 100}")
print(f"\n最终答案: {answer_content}")