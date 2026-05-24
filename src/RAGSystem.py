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
            السياق والنص المرجعي الحقيقي:
            \"\"\"
            {context}
            \"\"\"

            السؤال المطلوب الإجابة عنه:
            {query}
            """

        system_instruction = """
            أنت خبير لغوي مدقق ومتخصص في شرح وإعراب المعلقات والشعر العربي.
            مهمتك الصارمة هي الإجابة عن سؤال المستخدم بالاعتماد حصرًا وحرفيًا على "النص المرجعي" المقدم لك.

            قوانين عدم الهلوسة والالتزام (قواعد حازمة):
            1. استخرج الشرح، الإعراب، والمفردات من النص المرجعي المرفق فقط كما وردت فيه دون أي تعديل أو زيادة أو اجتهاد شخصي.
            2. يُمنع منعًا باتًا استخدام معرفتك الخارجية أو استنباط قواعد نحوية أو إعرابية أو صرفية من ذاكرتك العامة إذا لم تكن مذكورة نصًا في المرجع.
            3. إذا كان النص المرجعي يحتوي على قيم مثل "غير متوفر" أو لا يحتوي على تفاصيل الإعراب النحوي الدقيق للبيت المطلوب، فلا تقم بتأليف الإعراب من عندك، بل أجب للمستخدم حرفيًا بـ: "عذرًا، تفاصيل الإعراب لهذا البيت غير متوفرة في قاعدة البيانات".
            4. لا تقم بتحسين الإعراب أو تصحيحه لغويًا من ذاكرتك حتى لو اعتقدت أنه خاطئ؛ التزم بالأمانة العلمية لما هو مسترجع في السياق فقط.
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