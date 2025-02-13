from langchain_google_genai import ChatGoogleGenerativeAI
import os

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


# Chain-of-Thought (CoT) 
# Encourages the model to think step-by-step.
cot_prompt = """
Think step by step and explain your reasoning clearly.

Q1: A bakery bakes 120 cupcakes daily. If each customer buys 4 cupcakes, how many customers can be served in a day?

Q2: If the bakery increases production by 50%, how many total cupcakes will they bake per day?

Q3: Now, if each cupcake costs $2 and a customer buys 6 cupcakes, how much will the customer pay?

Q4: Assume the bakery spends $0.50 per cupcake on ingredients. If they bake 180 cupcakes in a day, what is their total cost for ingredients?
"""

# Invoke the model
response = model.invoke(cot_prompt)