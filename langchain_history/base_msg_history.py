import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_history.custom_msg_history import MyMsgHistory, FileChatMessageHistory
from models import get_ali_model_client

ali_model = get_ali_model_client()

# 定义提示模板，包含历史占位符
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个人工智能专家，简洁回答问题。"),
    # ("placeholder", "{messages}"),
    MessagesPlaceholder(variable_name="my_history"),
    ("human", "{input}")
])
chat_history = FileChatMessageHistory()
chat_history.storage_path = "my_history"
chat_history.session_id = f"message_{datetime.datetime.now().timestamp()}.json"
chain = prompt | ali_model
while True:
    input_str = input(f"\033[1;m{'you'}\033[0m: ")
    if input_str == "exit":
        print(chat_history.messages)
        break
    chat_history.add_user_message(input_str)
    stream = chain.stream({"input": input_str, "my_history": chat_history.messages})
    ai_message = ""
    for res in stream:
        ai_message+=res.content
        print(res.content)
    chat_history.add_ai_message(ai_message)
