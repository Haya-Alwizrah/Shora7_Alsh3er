from groq import Groq


class RAGSystem:

    def __init__(self, data_manager, groq_key, model_name):

        self.data_manager = data_manager
        self.client_g = Groq(api_key=groq_key)
        self.model_name = model_name

    def ask(self, query):
        docs = self.data_manager.search(query)
        context = "\n\n".join(docs)

        prompt = f"""
النص المرجعي:
{context}

السؤال:
{query}
"""

        response = self.client_g.chat.completions.create(
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