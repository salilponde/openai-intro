from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

OPENAI_ORG_ID = os.environ["OPENAI_ORG_ID"]
OPENAI_PROJECT_ID = os.environ["OPENAI_PROJECT_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.1

client = OpenAI(
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJECT_ID,
    api_key=OPENAI_API_KEY
)


class MessageHistory:
    def __init__(self, system_message: str, n: int) -> None:
        self.history = [{"role": "system", "content": system_message}]
        self.n = n

    def add_message(self, message_type, message_text):
        self.history.append({"role": message_type, "content": message_text})
        if len(self.history) > self.n:
            self.history.pop(1)

    def add_user_message(self, message):
        self.add_message("user", message)

    def add_ai_message(self, message):
        self.add_message("assistant", message)

    def get_messages(self) -> list[dict]:
        return self.history


message_history = MessageHistory(
    system_message="You are a helpful assistant.", n=5)


def call_ai(message):
    message_history.add_user_message(message=message)

    response = client.chat.completions.create(
        model=MODEL,
        messages=message_history.get_messages(),
        temperature=TEMPERATURE)

    ai_message = response.choices[0].message.content
    message_history.add_ai_message(ai_message)

    return ai_message


while True:
    user_message = input("You: ")
    if user_message == "quit":
        break
    ai_message = call_ai(user_message)
    print("AI :", ai_message)
