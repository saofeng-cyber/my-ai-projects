import uuid

from langchain_classic.retrievers import MultiVectorRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pydantic import BaseModel, Field

from models import get_ali_clients

# 获取大模型
# ollama_llm = get_ollama_completion()
# ali_embedding = get_ali_embeddings()
ollama_llm, ali_embedding = get_ali_clients()
# 加载文档
loader = TextLoader("../data/deepseek百度百科.txt", encoding="utf-8")
texts = loader.load()
# 分割器
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
split_texts = splitter.split_documents(texts)
prompt1 = ChatPromptTemplate.from_template(
    """
    基于以下内容生成假设性问题。
    重要：请直接返回 JSON 格式的数据，不要包含任何 Markdown 格式标记（如 ```json）或其他说明文字。
    内容：{doc}
    要求:
    1.输出内容必须为合法的json格式，包含questions字段
    2.questions字段的值是包含3个问题的数组
    3.使用中文提问
    示例格式:
    {{
        "questions": ["问题1", "问题2", "问题3"]
    }}
    """
)


# 总结：Field 是 Pydantic 中用于增强字段定义的工具，它让你能够：
#
# 标记字段是否必需
#
# 添加验证规则
#
# 提供文档描述
#
# 设置默认值
#
# 控制序列化行为
class HypotheticalQuestions(BaseModel):
    """生成假设性问题"""
    questions: list[str] = Field(..., description="List of questions")


# 使用结构化输出
chain = RunnableSequence({"doc": lambda x: x.page_content}, prompt1,
                         ollama_llm.with_structured_output(HypotheticalQuestions), lambda x: x.questions)
question_docs = chain.batch(split_texts)
print(question_docs)
id_key = "doc_id"
doc_ids = [str(uuid.uuid4()) for _ in split_texts]
question_document = []
for i, question_list in enumerate(question_docs):
    question_document.extend([Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list])
print("question_document", question_document)
vector_store = Chroma(collection_name="question_document", embedding_function=ali_embedding)
doc_store = InMemoryStore()
retriever = MultiVectorRetriever(vectorstore=vector_store, docstore=doc_store, id_key=id_key)
retriever.vectorstore.add_documents(question_document)
# mset批量设置键值对
retriever.docstore.mset(list(zip(doc_ids, split_texts)))

prompt = ChatPromptTemplate.from_template("请根据下面的文档内容, 回答问题: \n\n文档内容{doc}, \n\n问题{question}")
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
        print(chunk.content, end="", flush=True)

print(f"\n\n{'*' * 100}")
print(f"\n最终答案: {answer_content}")
