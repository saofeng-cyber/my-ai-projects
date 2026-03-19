import uuid

from langchain_classic.retrievers import MultiVectorRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from models import get_ali_embeddings, get_ollama_completion

# 获取大模型
ollama_llm = get_ollama_completion()
ali_embedding = get_ali_embeddings()
outParser = StrOutputParser()
# 加载文档
loader = TextLoader("../data/deepseek百度百科.txt", encoding="utf-8")
texts = loader.load()
# 分割器
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
split_texts = splitter.split_documents(texts)

chain = RunnableSequence({"doc": lambda x: x.page_content},
                         ChatPromptTemplate.from_template("为下面这段内容做一个关键总结: {doc}"), ollama_llm, outParser)
summary_docs = chain.batch(split_texts)
id_key = "doc_id"
doc_ids = [str(uuid.uuid4()) for _ in summary_docs]
summary_document = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(summary_docs)]
print("summary_document", summary_document)
vector_store = Chroma(collection_name="summaries", embedding_function=ali_embedding)
doc_store = InMemoryStore()
retriever = MultiVectorRetriever(vectorstore=vector_store, docstore=doc_store, id_key=id_key)
retriever.vectorstore.add_documents(summary_document)
# mset批量设置键值对
retriever.docstore.mset(list(zip(doc_ids, split_texts)))

prompt = ChatPromptTemplate.from_template("请根据下面的文档内容和问题进行回答: \n\n文档内容{doc}, \n\n问题{question}")
llm_chain = RunnableParallel(
    {"doc": lambda x: retriever.invoke(x["question"]), "question": lambda x: x["question"]}) | prompt | ollama_llm
stream = llm_chain.stream({"question": "deepseek大模型的早期发展"})

# 存储思考过程和最终答案
reasoning_content = ""
answer_content = ""
for chunk in stream:
    if hasattr(chunk, "additional_kwargs") and 'reasoning_content' in chunk.additional_kwargs:
        reasoning_part = chunk.additional_kwargs['reasoning_content']
        if reasoning_part:
            reasoning_part += reasoning_part
            print(reasoning_part, end="", flush=True)
    if chunk.content:
        answer_content += chunk.content
        print(answer_content, end="", flush=True)

print(f"\n\n{'*' * 100}")
print(f"\n最终答案: {answer_content}")
