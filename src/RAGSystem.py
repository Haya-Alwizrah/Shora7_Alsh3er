# from groq import Groq
import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(".env")

class RAGSystem:


    def __init__(self, data_manager, OPENAI_API_KEY, OPENAI_MODEL):
        self.data_manager = data_manager
        
        # self.client_g = Groq(api_key=OPENAI_API_KEY)
        # self.model_name = model_name

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # self.client = OpenAI(api_key=self.api_key)
        
    def ask(self, query):
        docs = self.data_manager.search(query)
        context = "\n\n".join(docs)

        prompt = f"""
النص المرجعي:
{context}

السؤال:
{query}
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content":
                    "أنت مساعد متخصص في إعراب وشرح الشعر العربي. أجب اعتمادًا على المعلومات المعطاة فقط."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content.strip()

        return answer, context