# generator.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def generate_answer(context, question):
    prompt = f"""You are an assistant that answers questions based on event brochures.

Context:
{context}

Question:
{question}

Answer:"""

    response = model.generate_content(prompt)
    return response.text.strip()
