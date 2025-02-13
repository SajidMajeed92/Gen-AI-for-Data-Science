from langchain_google_genai import ChatGoogleGenerativeAI
import os

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


# Gives explicit instructions to control response format.

instruction_prompt = """
Write a short, enthusiastic email (max 4 lines) applying for the AI Engineer role at Samsung. Highlight AI skills.
"""



response = model.invoke(instruction_prompt)
print(response.content)