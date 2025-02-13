from langchain_google_genai import ChatGoogleGenerativeAI
import os

# call model and API Keys from .env file 
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key= os.environ["GOOGLE_API_KEY"])   

result = model.invoke("what is square root of 8")# invoke magic function

print(result)

print(result.content)