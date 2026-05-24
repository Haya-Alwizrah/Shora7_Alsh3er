# from groq import Groq
import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(".env")

class RAGSystem:


    def __init__(self, data_manager, OPENAI_API_KEY, OPENAI_MODEL):
        self.data_manager = data_manager

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
    def ask(self, query):
        context = self.data_manager.search(query)
        #context = "\n\n".join(docs)

        prompt = f"""
            النص المرجعي:
            \"\"\"
            {context}
            \"\"\"

            السؤال المطلوب الإجابة عنه:
            {query}
            """

        system_instruction = """
            أنت خبير لغوي مدقق ومتخصص في شرح مفردات ومعاني وإعراب المعلقات العشر.
            مهمتك الصارمة هي الإجابة عن سؤال المستخدم بالاعتماد حصرًا وحرفيًا على "النص المرجعي" المقدم لك.

            قوانين عدم الهلوسة والالتزام (قواعد حازمة):
            1. استخرج الجواب من النص المرجعي المرفق فقط كما ورد دون أي تعديل أو زيادة أو اجتهاد شخصي.
            2. يُمنع منعًا باتًا استخدام معرفتك الخارجية أو استنباط معاني مفردات أو شرح للبيت أو قواعد (نحوية، إعرابية، أو صرفية) من ذاكرتك العامة إذا لم تكن مذكورة نصًا في المرجع.
            3. إذا كان النص المرجعي لا يحتوي على جواب للسؤال المطلوب، فلا تقم باختلاق جواب من عندك، بل أجب للمستخدم حرفيًا بـ: "عذرًا، تفاصيل الجواب لهذا البيت غير متوفرة في قاعدة البيانات".
            4. لا تقم بتحسين الجواب أو تصحيحه من ذاكرتك حتى لو اعتقدت أنه خاطئ؛ التزم بالأمانة العلمية لما هو مسترجع في السياق فقط.
            5. لا تقم باختصار أي جواب اعطي الشرح كاملاً.
            """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_instruction
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