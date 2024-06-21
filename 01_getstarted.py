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


def call_ai(message):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ], temperature=TEMPERATURE)

    return response.choices[0].message.content


while True:
    user_message = input("You: ")
    if user_message == "quit":
        break
    ai_message = call_ai(user_message)
    print("AI :", ai_message)
