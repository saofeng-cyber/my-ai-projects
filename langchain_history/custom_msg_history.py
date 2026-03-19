import json
import os
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict, AIMessage
from pydantic import BaseModel, Field

# 内存
class MyMsgHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)


    def add_ai_message(self, message: AIMessage | str) -> None:
        self.messages.append(message)

    def clear(self) -> None:
        self.messages.clear()


class FileChatMessageHistory(BaseChatMessageHistory):
    storage_path: str
    session_id: str

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(
                    os.path.join(self.storage_path, self.session_id),
                    "r",
                    encoding="utf-8",
            ) as f:
                messages_data = json.load(f)
            return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages)  # Existing messages
        all_messages.extend(messages)  # Add new messages

        serialized = [message_to_dict(message) for message in all_messages]
        file_path = os.path.join(self.storage_path, self.session_id)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False)

    def clear(self) -> None:
        file_path = os.path.join(self.storage_path, self.session_id)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
