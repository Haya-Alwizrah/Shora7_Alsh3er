EMBED_MODEL = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_KEY = "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"

DATA_FOLDER = "datasets"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "poetry"

# --------------------------------------------------------
from DataManager import DataManager
data_manager = DataManager(
    folder_path=DATA_FOLDER,
    db_path=CHROMA_PATH,
    collection_name=COLLECTION_NAME,
    embed_model=EMBED_MODEL
)

data_manager.prepare()

#--------------------------------------------------------
from RAGSystem import RAGSystem
rag_system = RAGSystem(
    data_manager=data_manager,
    groq_key=GROQ_KEY,
    model_name=GROQ_MODEL
)

question = "ما اعراب ودع هريرة ان الرحل مرتحل"
answer, context = rag_system.ask(question)
print(answer)

# ---------------------------------------------
from LLMJudge import LLMJudge
judge_system = LLMJudge(
    groq_key=GROQ_KEY,
    model_name=GROQ_MODEL
)

judge_result = judge_system.judge(question, answer, context)
print(judge_result)

#------------------------------------------------------------------------
from RAGEvaluator import RAGEvaluator
evaluator = RAGEvaluator(
    groq_key=GROQ_KEY,
    model_name=GROQ_MODEL,
    embed_model=EMBED_MODEL
)

eval_data = evaluator.load_from_excel("eval_data.xlsx")
dataset = evaluator.build_dataset(rag_system, eval_data)
result = evaluator.evaluate(dataset)
print(result)