from langchain_google_genai import ChatGoogleGenerativeAI
import os

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


# Assigns a role to the model to influence its response style.
role_based_prompt = """
You are an expert AI recruiter. Write a compelling email to Samsung about my interest in the AI Engineer position.
"""

response = model.invoke(role_based_prompt)
print(response.content)