import streamlit as st
import pandas as pd
import os
import warnings
from dotenv import load_dotenv
import sys
from types import ModuleType

warnings.filterwarnings("ignore")

from src.DataManager2 import DataManager2
from src.RAGSystem import RAGSystem
from src.RagGraph import GraphRag
from src.RAGEvaluationSystem import RAGEvaluationSystem

from views import CSS_STYLE, render_home, render_evaluation, render_details

load_dotenv()

EMBED_MODEL    = os.getenv("EMBEDDING_MODEL2", "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY")
DATASET        = os.getenv("DATASET", "SarahALo/The-Ten-Muallaqat-Dataset")
CHROMA_PATH     = "cache/chroma_db"
COLLECTION_NAME = "muallaqat_collection"
EVAL_PATH       = "cache/eval_data.xlsx"

if 'langchain_community.chat_models.vertexai' not in sys.modules:
    mock_vertex = ModuleType('langchain_community.chat_models.vertexai')
    mock_vertex.ChatVertexAI = None
    sys.modules['langchain_community.chat_models.vertexai'] = mock_vertex

if 'langchain_community.llms' not in sys.modules:
    mock_llm = ModuleType('langchain_community.llms')
    mock_llm.VertexAI = None
    sys.modules['langchain_community.llms'] = mock_llm
    
st.set_page_config(page_title="شُرّاح الشعر", page_icon="📜", layout="wide")

st.markdown(CSS_STYLE, unsafe_allow_html=True)

@st.cache_resource
def init_systems():
    dm = DataManager2(
        dataset_name=DATASET, 
        db_path=CHROMA_PATH, 
        collection_name=COLLECTION_NAME, 
        EMBEDDING_MODEL=EMBED_MODEL
    )
    dm.prepare()
    standard_rag = RAGSystem(dm)
    graph_rag    = GraphRag()
    evaluator    = RAGEvaluationSystem(OPENAI_KEY, OPENAI_MODEL, EMBED_MODEL)
    return standard_rag, graph_rag, evaluator

try:
    std_rag, g_rag, eval_sys = init_systems()
except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل الأنظمة: {e}")
    st.stop()

if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None

st.sidebar.markdown("""
<div style='text-align:center;padding:6px 0 18px;border-bottom:0.5px solid rgba(255,255,255,0.12);margin-bottom:12px'>
  <div style='font-size:30px;margin-bottom:4px'>📜</div>
  <h2 style='font-family:Amiri,serif;font-size:20px;color:#F5E6B8;margin:6px 0 3px'>شُرّاح الشعر</h2>
  <p style='font-size:10.5px;color:rgba(255,255,255,0.4);margin:0'>تحليل المعلقات العشر </p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    [
        "🏠 الرئيسية",
        "◈  لوحة التقييم",
        "📋  تفاصيل المشروع",
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("""
<div style='margin-top:20px;padding-top:16px;border-top:0.5px solid rgba(255,255,255,0.1);
     font-size:10px;color:rgba(255,255,255,0.28);text-align:center;line-height:2'>
  <br>
  <span style='color:rgba(255,255,255,0.18)'>Haya Alwizrah · Sarah ALowjan</span>
</div>
""", unsafe_allow_html=True)

if menu == "🏠 الرئيسية":
    render_home(std_rag, g_rag)

elif menu == "◈  لوحة التقييم":
    render_evaluation(eval_sys, std_rag, g_rag, EVAL_PATH)

elif menu == "📋  تفاصيل المشروع":
    render_details()