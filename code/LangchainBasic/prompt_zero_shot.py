from langchain_google_genai import ChatGoogleGenerativeAI
import os

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Zero-shot prompt
# Directly asks the model to complete a task without any examples.

zero_shot = "Write a professional email expressing interest in the AI Engineer position at Samsung."

# Generate response
response = model.invoke(zero_shot)
print(response.content)


