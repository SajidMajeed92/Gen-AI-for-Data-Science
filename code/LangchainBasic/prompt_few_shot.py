from langchain_google_genai import ChatGoogleGenerativeAI
import os

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


# Provides a few examples to guide the model’s response

few_shot_prompt = """
Example 1:
Subject: Excited to Apply for AI Engineer Role at Samsung
Hi Hiring Manager,
I am excited to apply for the AI Engineer position at Samsung. With expertise in AI and ML, I believe I can contribute significantly. Looking forward to the opportunity.
Best, 
John Doe

Now, write a similar email for a Data Scientist role at Google.
"""

response = model.invoke(few_shot_prompt)
print(response.content)