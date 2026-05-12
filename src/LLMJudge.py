from groq import Groq

class LLMJudge:

    def __init__(self, groq_key, model_name):

        self.client_g = Groq(api_key=groq_key)
        self.model_name = model_name

    def judge(self, question, answer, context):

        prompt = f"""
أنت مقيم دقيق لإجابات نظام RAG.

قيّم الإجابة من 1 إلى 100 في:
- الدقة
- الالتزام بالسياق
- مدى ارتباطها بالسؤال

أعد النتيجة بصيغة JSON فقط.

---

السؤال:
{question}

---

السياق:
{context}

---

الإجابة:
{answer}
"""

        response = self.client_g.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "أنت حكم محايد ودقيق."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content