from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd

class RAGEvaluator:

    def __init__(self, groq_key, model_name, embed_model):

        self.evaluator_llm = ChatGroq(
            groq_api_key=groq_key,
            model_name=model_name,
            temperature=0,
            max_tokens=1024,
            n=1
        )

        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)

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

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })

        return dataset
    
    def load_from_excel(self, file_path):

        df = pd.read_excel(file_path)

        eval_data = []

        for _, row in df.iterrows():

            eval_data.append({
                "question": row["question"],
                "ground_truth": row["ground_truth"]
            })

        return eval_data
    
    def evaluate(self, dataset):

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