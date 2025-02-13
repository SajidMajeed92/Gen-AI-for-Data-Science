from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
import os

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

messages = [
    SystemMessage(content="You are an expert in social media content strategy"),
    HumanMessage(content="Give a short tip to create engaging posts on Instagram"),
    ]
result = model.invoke(messages)
print(f"Answer from Google: {result.content}")