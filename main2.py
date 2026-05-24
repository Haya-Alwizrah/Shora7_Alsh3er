import os
from dotenv import load_dotenv
from src.DataManager import DataManager
from src.RAGSystem import RAGSystem
from src.RAGEvaluationSystem import RAGEvaluationSystem
load_dotenv()

EMBED_MODEL = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
#EMBED_MODEL     = os.getenv("EMBEDDING_MODEL", "Sarah0001/Arabic_embed_model")
DATASET         = os.getenv("DATASET", "SarahALo/The-Ten-Muallaqat-Dataset")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL",    "gpt-4o-mini")
OPENAI_KEY      = os.getenv("OPENAI_API_KEY")
CHROMA_PATH     = "cache/chroma_db"
COLLECTION_NAME = "muallaqat_collection"
EVAL_PATH       = "cache/eval_data.xlsx"

# --------------------------------------------------------
data_manager = DataManager(
    dataset_name=DATASET,  
    db_path=CHROMA_PATH,
    collection_name=COLLECTION_NAME,
    EMBEDDING_MODEL=EMBED_MODEL
)
data_manager.prepare()
#--------------------------------------------------------
from src.RAGSystem import RAGSystem
rag_system = RAGSystem( data_manager, OPENAI_KEY, OPENAI_MODEL)

question = "ما معنى تضيء الظلام العشاء كأنها"
answer, context = rag_system.ask(question)
print(answer)
print(context)

# ---------------------------------------------
from src.RAGEvaluationSystem import RAGEvaluationSystem
evaluator = RAGEvaluationSystem(OPENAI_KEY, OPENAI_MODEL, EMBED_MODEL)

judge_result = evaluator.judge(question, answer, context)
print(judge_result)

#result = evaluator.ragas_eval(rag_system, EVAL_PATH)
#print(result)
