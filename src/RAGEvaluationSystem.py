import pandas as pd
from datasets import Dataset

from groq import Groq

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

class RAGEvaluationSystem:
    def __init__(self, groq_key, model_name, embed_model):
        
        # RAGAS:
        self.evaluator_llm = ChatGroq(
            groq_api_key=groq_key,
            model_name=model_name,
            temperature=0,
            max_tokens=1024,
            n=1
        )

        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)

        # LLM Judge:
        self.client = Groq(api_key=groq_key)
        self.model_name = model_name

# ----------------------------------------------------------------------------------------
    def load_from_excel(self, file_path):
        df = pd.read_excel(file_path)

        eval_data = []

        for _, row in df.iterrows():

            eval_data.append({
                "question": row["question"],
                "ground_truth": row["ground_truth"]
            })

        return eval_data
    
    def build_dataset(self, rag_system, eval_data):

        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for item in eval_data:

            q = item["question"]

            answer, context = rag_system.ask(q)

            questions.append(q)
            answers.append(answer)
            contexts.append([context])
            ground_truths.append(item["ground_truth"])

        return Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })
    
    def evaluate_ragas(self, dataset):

        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(),
                AnswerRelevancy(),
                ContextPrecision(),
                ContextRecall()
            ],
            llm=self.evaluator_llm,
            embeddings=self.embeddings
        )

        return result
    
    def ragas_eval(self, rag_system, excel_path):

        eval_data = self.load_from_excel(excel_path)
        dataset = self.build_dataset(rag_system, eval_data)
        result = self.evaluate_ragas(dataset)

        return result  

# -----------------------------------------------------------------------------------      
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

        response = self.client.chat.completions.create(
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