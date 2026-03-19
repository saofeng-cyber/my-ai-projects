from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig

from models import get_ali_clients

ali_model, ali_embeddings = get_ali_clients()

# 历史记忆
# 定义提示模板，包含历史占位符
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个人工智能专家，简洁回答问题。"),
    MessagesPlaceholder(variable_name="my_history"),
    ("human", "{input}")
])
chain = prompt | ali_model

store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(runnable=chain, get_session_history= get_session_history,
                                                  input_messages_key="input", history_messages_key="my_history")
while True:
    input_str = input(f"\033[1;m{'you'}\033[0m: ")
    if input_str == "exit":
        print(get_session_history("my_history").messages)
        break
    stream = with_message_history.stream(input={"input": input_str},
                                         config=RunnableConfig(configurable={"session_id": "my_history"}))
    result_content: str = ''
    for s in stream:
        if s.content:
            result_content += s.content
            print(s.content, sep="", end="", flush=True)
    print("\n")
