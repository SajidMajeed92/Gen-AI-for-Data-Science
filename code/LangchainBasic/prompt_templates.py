from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max"

prompt_template = ChatPromptTemplate.from_template(template)

prompt =  prompt_template.invoke({
"tone": "energetic", 
"company": "samsung", 
"position": "AI Engineer", 
"skill": "AI"
})
result = model.invoke(prompt)
print(f"Answer from Google: {result.content}")