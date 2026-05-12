import streamlit as st

files = [

    "Dataset/معلقة الأعشى.txt",

    "Dataset/معلقة عنترة بن شداد.txt",

    "Dataset/معلقة لبيد بن ربيعة.txt"
]

EMBED_MODEL = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_KEY = "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"

DATA_FOLDER = "Dataset"
EVAL_PATH = "src/eval_data.xlsx"
CHROMA_PATH = "cache/chroma_db"
COLLECTION_NAME = "poetry"

# --------------------------------------------------------
from src.DataManager import DataManager
from src.RAGSystem import RAGSystem
from src.RAGEvaluationSystem import RAGEvaluationSystem

#---------------------------------------------------
st.set_page_config(
    page_title="Arabic Poetry RAG",
    page_icon="📖",
    layout="wide"
)

st.title("📖 Arabic Poetry RAG System")
# ---------------------------------------
@st.cache_resource
def load_system():

    data_manager = DataManager(
        files=files,
        db_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embed_model=EMBED_MODEL
    )

    data_manager.prepare()

    rag_system = RAGSystem(
        data_manager=data_manager,
        groq_key=GROQ_KEY,
        model_name=GROQ_MODEL
    )

    evaluator = RAGEvaluationSystem(
        groq_key=GROQ_KEY,
        model_name=GROQ_MODEL,
        embed_model=EMBED_MODEL
    )

    return rag_system, evaluator

rag_system, evaluator = load_system()

question = st.text_input(
    "اكتب سؤالك عن احد المعلقات العشر:",
    placeholder="مثال: ما اعراب ودع هريرة ان الركب مرتحل"
)


if st.button("اسأل"):

    if question.strip() == "":
        st.warning("اكتب سؤال أول")
    else:

        with st.spinner("جاري توليد الإجابة..."):

            answer, context = rag_system.ask(question)

        # ------------------------------------------------
        # ANSWER
        # ------------------------------------------------
        st.subheader("📌 الإجابة")
        st.write(answer)

        # ------------------------------------------------
        # JUDGE
        # ------------------------------------------------
        with st.spinner("جاري تقييم الإجابة..."):

            judge_result = evaluator.judge(
                question,
                answer,
                context
            )

        st.subheader("⚖️ تقييم الـ LLM Judge")
        st.write(judge_result)

st.divider()

st.subheader("📊 تقييم النظام بالكامل")
