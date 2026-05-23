import os
from dotenv import load_dotenv
from src.DataManager import DataManager
from src.RAGSystem import RAGSystem
from src.RAGEvaluationSystem import RAGEvaluationSystem

EMBED_MODEL = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_KEY = "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"

DATA_FOLDER = "Dataset"
EVAL_PATH = "cache/eval_data.xlsx"
CHROMA_PATH = "cache/chroma_db"
COLLECTION_NAME = "poetry"

load_dotenv()

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "Sarah0001/Arabic_embed_model")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")


# --------------------------------------------------------
data_manager = DataManager(
    dataset_name="SarahALo/The-Ten-Muallaqat-Dataset",  
    db_path=CHROMA_PATH,
    collection_name="muallaqat_collection",
    EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL")
)
data_manager.prepare()
#--------------------------------------------------------
from src.RAGSystem import RAGSystem
rag_system = RAGSystem(
    data_manager, os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MODEL")
)

question = "تذكرت الصبا، واشتقت لما"
answer, context = rag_system.ask(question)
print(answer)

# ---------------------------------------------
from src.RAGEvaluationSystem import RAGEvaluationSystem
evaluator = RAGEvaluationSystem(os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MODEL"), EMBED_MODEL)


judge_result = evaluator.judge(question, answer, context)
print(judge_result)

#result = evaluator.ragas_eval(rag_system, EVAL_PATH)
#print(result)