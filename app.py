import streamlit as st
import pandas as pd
import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

from src.DataManager import DataManager
from src.RAGSystem import RAGSystem
from src.RagGraph import GraphRag
from src.RAGEvaluationSystem import RAGEvaluationSystem

load_dotenv()

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "Sarah0001/Arabic_embed_model")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

DATA_FOLDER = "Dataset"
CHROMA_PATH = "cache/chroma_db"
COLLECTION_NAME = "poetry"
EVAL_PATH = "cache/eval_data.xlsx"

# اعدادات الصفحه 
st.set_page_config(
    page_title="شُرّاح الشعر",
    page_icon="📜",
    layout="wide"
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=IBM+Plex+Sans+Arabic:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans Arabic', sans-serif !important;
    direction: rtl;
}

.stApp { background-color: #FAF6ED !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #5B2D0E 0%, #3D1A05 100%) !important;
}
section[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.82) !important;
    font-family: 'IBM Plex Sans Arabic', sans-serif !important;
}
section[data-testid="stSidebar"] .stRadio label {
    padding: 9px 14px !important;
    border-radius: 10px !important;
    margin: 2px 0 !important;
    font-size: 14px !important;
    transition: background .2s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.08) !important;
}

.shurah-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 24px; padding-bottom: 16px;
    border-bottom: 1px solid #E2D9C4;
}
.shurah-header h1 {
    font-family: 'Amiri', serif !important;
    font-size: 26px; color: #5B2D0E; font-weight: 700; margin: 0;
}
.shurah-header p { font-size: 12.5px; color: #8A7D65; margin: 4px 0 0; }

.stat-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin-bottom: 20px;
}
.stat-card {
    background: white; border: 0.5px solid #E2D9C4;
    border-radius: 12px; padding: 16px 14px;
    transition: box-shadow .2s, transform .15s;
}
.stat-card:hover { box-shadow: 0 2px 14px rgba(0,0,0,0.08); transform: translateY(-1px); }
.stat-card .icon { font-size: 18px; margin-bottom: 8px; display: block; opacity: 0.8; }
.stat-card .val  { font-family: 'Amiri', serif; font-size: 24px; font-weight: 700; color: #5B2D0E; display: block; line-height: 1; }
.stat-card .lbl  { font-size: 11px; color: #9A8F78; margin-top: 4px; display: block; }

.content-card {
    background: white; border: 0.5px solid #E2D9C4;
    border-radius: 12px; padding: 20px 22px; margin-bottom: 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.content-card h3  { font-size: 14px; font-weight: 600; color: #1A1208; margin: 0 0 3px; }
.content-card .sub { font-size: 11.5px; color: #9A8F78; margin: 0 0 14px; }

.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

.cmp-wrap { display: flex; gap: 10px; }
.cmp-col  { flex: 1; border: 0.5px solid #E2D9C4; border-radius: 10px; overflow: hidden; }
.cmp-head { padding: 10px 14px; font-size: 12.5px; font-weight: 600; }
.cmp-head.std  { background: #EDE8F5; color: #26215C; }
.cmp-head.grph { background: #E1F5EE; color: #04342C; }
.cmp-row { display: flex; align-items: flex-start; gap: 7px; padding: 8px 14px; border-top: 0.5px solid #F0EAD8; font-size: 11.5px; color: #3A2C20; line-height: 1.6; }
.cmp-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; margin-top: 6px; }

.stButton > button {
    background: #5B2D0E !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.55rem 1.6rem !important; font-size: 14px !important;
    font-weight: 500 !important;
    font-family: 'IBM Plex Sans Arabic', sans-serif !important;
    transition: all .2s !important;
    box-shadow: 0 2px 8px rgba(91,45,14,0.22) !important;
}
.stButton > button:hover {
    background: #3D1A05 !important; transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(91,45,14,0.28) !important;
}

.stTextInput > div > div > input {
    border-radius: 10px !important; border: 1px solid #DDD5B8 !important;
    padding: 12px 16px !important; font-size: 15px !important;
    font-family: 'IBM Plex Sans Arabic', sans-serif !important;
    background: white !important; transition: border .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #5B2D0E !important;
    box-shadow: 0 0 0 3px rgba(91,45,14,0.1) !important;
}

.stChatMessage {
    border-radius: 12px !important; border: 0.5px solid #E2D9C4 !important;
    margin-bottom: 12px !important; background: white !important; padding: 14px !important;
}

[data-testid="stTable"] {
    background: white !important; border-radius: 12px !important;
    border: 0.5px solid #E2D9C4 !important; overflow: hidden !important;
}

.streamlit-expanderHeader {
    border-radius: 10px !important; background: #F2ECD8 !important;
    font-size: 13px !important; font-family: 'IBM Plex Sans Arabic', sans-serif !important;
}

.badge-std  { display:inline-block; background:#EDE8F5; color:#26215C; font-size:12px; padding:3px 12px; border-radius:20px; font-weight:500; }
.badge-grph { display:inline-block; background:#E1F5EE; color:#04342C; font-size:12px; padding:3px 12px; border-radius:20px; font-weight:500; }

.dev-row { display:flex; align-items:center; gap:10px; padding:9px 0; border-bottom:0.5px solid #F0EAD8; }
.dev-row:last-child { border-bottom:none; }
.avatar { width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:600; flex-shrink:0; }
.av1 { background:#EDE8F5; color:#3C3489; }
.av2 { background:#E1F5EE; color:#085041; }
.av3 { background:#FEF4E0; color:#633806; }
.dev-name { font-size:13px; font-weight:500; color:#1A1208; }
.dev-sub  { font-size:11px; color:#9A8F78; margin-top:1px; }

.tech-row { display:flex; justify-content:space-between; align-items:flex-start; padding:8px 0; border-bottom:0.5px solid #F0EAD8; font-size:12px; }
.tech-row:last-child { border-bottom:none; }
.tech-lbl { color:#6A6050; flex-shrink:0; }
.tech-val { font-weight:500; font-size:11.5px; text-align:left; }

.footer-note {
    margin-top:12px; padding:12px 16px;
    background:#F2ECD8; border-radius:10px;
    font-size:11px; color:#7A6A50;
    text-align:center; line-height:1.7;
}

hr { border:none !important; border-top:0.5px solid #E2D9C4 !important; margin:20px 0 !important; }
</style>
""", unsafe_allow_html=True)

# Knowledge Graph SVG
KG_SVG = """
<div class="content-card">
  <h3>قاعدة البيانات المعرفية — Neo4j Knowledge Graph</h3>
  <p class="sub">1,289 عقدة · 1,023 علاقة · خاصيتان: <code>embedding</code> و <code>text</code></p>
  <div style="display:flex;gap:16px;margin-bottom:12px;flex-wrap:wrap">
    <span style="display:flex;align-items:center;gap:6px;font-size:11.5px;color:#3A2C20">
      <span style="width:10px;height:10px;border-radius:50%;background:#534AB7;display:inline-block"></span>Verse (250)
    </span>
    <span style="display:flex;align-items:center;gap:6px;font-size:11.5px;color:#3A2C20">
      <span style="width:10px;height:10px;border-radius:50%;background:#0F6E56;display:inline-block"></span>Meaning (250)
    </span>
    <span style="display:flex;align-items:center;gap:6px;font-size:11.5px;color:#3A2C20">
      <span style="width:10px;height:10px;border-radius:50%;background:#854F0B;display:inline-block"></span>Grammar (250)
    </span>
    <span style="display:flex;align-items:center;gap:6px;font-size:11.5px;color:#3A2C20">
      <span style="width:10px;height:10px;border-radius:50%;background:#185FA5;display:inline-block"></span>Vocabulary (250)
    </span>
    <span style="display:flex;align-items:center;gap:6px;font-size:11.5px;color:#3A2C20">
      <span style="width:10px;height:10px;border-radius:50%;background:#888;display:inline-block"></span>أخرى (289)
    </span>
  </div>
  <svg width="100%" viewBox="0 0 680 200" style="font-family:'IBM Plex Sans Arabic',sans-serif">
    <defs>
      <marker id="arkg" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </marker>
    </defs>
    <rect width="680" height="200" rx="10" fill="#FAFAF7"/>
    <line x1="130" y1="88" x2="235" y2="65" stroke="#534AB7" stroke-width="1.2" stroke-opacity="0.5" marker-end="url(#arkg)" fill="none"/>
    <line x1="130" y1="112" x2="235" y2="138" stroke="#534AB7" stroke-width="1" stroke-opacity="0.4" marker-end="url(#arkg)" fill="none"/>
    <line x1="298" y1="60" x2="400" y2="80" stroke="#0F6E56" stroke-width="1" stroke-opacity="0.45" marker-end="url(#arkg)" fill="none"/>
    <line x1="298" y1="143" x2="400" y2="118" stroke="#854F0B" stroke-width="1" stroke-opacity="0.4" marker-end="url(#arkg)" fill="none"/>
    <line x1="465" y1="88" x2="545" y2="96" stroke="#185FA5" stroke-width="0.8" stroke-opacity="0.35" marker-end="url(#arkg)" fill="none"/>
    <text x="182" y="63" text-anchor="middle" font-size="9" fill="#9A8F78" font-family="IBM Plex Sans Arabic">HAS_MEANING</text>
    <text x="182" y="140" text-anchor="middle" font-size="9" fill="#9A8F78" font-family="IBM Plex Sans Arabic">HAS_GRAMMAR</text>
    <text x="350" y="84" text-anchor="middle" font-size="9" fill="#9A8F78" font-family="IBM Plex Sans Arabic">HAS_VOCABULARY</text>
    <circle cx="90" cy="100" r="40" fill="#EDE8F5" stroke="#534AB7" stroke-width="1.5"/>
    <text x="90" y="95" text-anchor="middle" font-size="12" fill="#26215C" font-weight="600" font-family="IBM Plex Sans Arabic">Verse</text>
    <text x="90" y="110" text-anchor="middle" font-size="10" fill="#534AB7" font-family="IBM Plex Sans Arabic">250 عقدة</text>
    <circle cx="268" cy="58" r="32" fill="#E1F5EE" stroke="#0F6E56" stroke-width="1.5"/>
    <text x="268" y="53" text-anchor="middle" font-size="11" fill="#04342C" font-weight="600" font-family="IBM Plex Sans Arabic">Meaning</text>
    <text x="268" y="68" text-anchor="middle" font-size="10" fill="#0F6E56" font-family="IBM Plex Sans Arabic">250 عقدة</text>
    <circle cx="268" cy="148" r="32" fill="#FAEEDA" stroke="#854F0B" stroke-width="1.5"/>
    <text x="268" y="143" text-anchor="middle" font-size="11" fill="#412402" font-weight="600" font-family="IBM Plex Sans Arabic">Grammar</text>
    <text x="268" y="158" text-anchor="middle" font-size="10" fill="#854F0B" font-family="IBM Plex Sans Arabic">250 عقدة</text>
    <circle cx="435" cy="100" r="34" fill="#E6F1FB" stroke="#185FA5" stroke-width="1.5"/>
    <text x="435" y="95" text-anchor="middle" font-size="10.5" fill="#042C53" font-weight="600" font-family="IBM Plex Sans Arabic">Vocabulary</text>
    <text x="435" y="110" text-anchor="middle" font-size="10" fill="#185FA5" font-family="IBM Plex Sans Arabic">250 عقدة</text>
    <circle cx="575" cy="100" r="24" fill="#F1EFE8" stroke="#888780" stroke-width="1"/>
    <text x="575" y="96" text-anchor="middle" font-size="10" fill="#444441" font-family="IBM Plex Sans Arabic">أخرى</text>
    <text x="575" y="110" text-anchor="middle" font-size="10" fill="#5F5E5A" font-family="IBM Plex Sans Arabic">289</text>
    <text x="340" y="190" text-anchor="middle" font-size="9.5" fill="#B0A898" font-family="IBM Plex Sans Arabic">
      إجمالي العلاقات: 1,023 · HAS_MEANING · HAS_GRAMMAR · HAS_VOCABULARY
    </text>
  </svg>
</div>
"""

COMPARE_HTML = """
<div class="content-card">
  <h3>المقارنة بين النظامين</h3>
  <p class="sub">ما الفرق بين Standard RAG والـ Graph RAG؟</p>
  <div class="cmp-wrap">
    <div class="cmp-col">
      <div class="cmp-head std">◎ Standard RAG — البحث التقليدي</div>
      <div class="cmp-row"><span class="cmp-dot" style="background:#534AB7"></span>يُقسّم النصوص إلى chunks ويحوّلها إلى vectors</div>
      <div class="cmp-row"><span class="cmp-dot" style="background:#534AB7"></span>يبحث في ChromaDB بالتشابه الدلالي (cosine similarity)</div>
      <div class="cmp-row"><span class="cmp-dot" style="background:#534AB7"></span>سريع ومناسب للأسئلة المباشرة عن المعنى</div>
      <div class="cmp-row"><span class="cmp-dot" style="background:#534AB7"></span>لا يدرك الروابط والعلاقات بين الأبيات</div>
      <div class="cmp-row"><span class="cmp-dot" style="background:#534AB7"></span>Embedding: Sarah0001/Arabic_embed_model</div>
    </div>
    <div class="cmp-col">
      <div class="cmp-head grph">⬡ Graph RAG — البحث المعرفي</div>
      <div class="cmp-row"><span class="cmp-dot" style="background:#0F6E56"></span>يخزّن البيانات كعقد وعلاقات في Neo4j</div>
      <div class="cmp-row"><span class="cmp-dot" style="background:#0F6E56"></span>يربط: Verse ← Meaning ← Grammar ← Vocabulary</div>
      <div class="cmp-row"><span class="cmp-dot" style="background:#0F6E56"></span>يُجيب على الأسئلة التحليلية والمعرفية المعقّدة</div>
      <div class="cmp-row"><span class="cmp-dot" style="background:#0F6E56"></span>يستوعب السياق والترابط بين العناصر الشعرية</div>
      <div class="cmp-row"><span class="cmp-dot" style="background:#0F6E56"></span>أدق في الإعراب والتحليل النحوي التفصيلي</div>
    </div>
  </div>
</div>
"""

# تهيئة الأنظمة
@st.cache_resource
def init_systems():
    dm = DataManager(DATA_FOLDER, CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL)
    dm.prepare()
    standard_rag = RAGSystem(dm, os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MODEL"))
    graph_rag = GraphRag()
    evaluator = RAGEvaluationSystem(
        os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MODEL"), EMBED_MODEL
    )
    return standard_rag, graph_rag, evaluator

try:
    std_rag, g_rag, eval_sys = init_systems()
except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل الأنظمة: {e}")
    st.stop()

if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None

# القائمة الجانبية
st.sidebar.markdown("""
<div style='text-align:center;padding:6px 0 18px;border-bottom:0.5px solid rgba(255,255,255,0.12);margin-bottom:12px'>
  <div style='font-size:30px;margin-bottom:4px'>📜</div>
  <h2 style='font-family:Amiri,serif;font-size:20px;color:#F5E6B8;margin:6px 0 3px'>شُرّاح الشعر</h2>
  <p style='font-size:10.5px;color:rgba(255,255,255,0.4);margin:0'>تحليل المعلقات العربية</p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    [
        "🏠  الرئيسية",
        "◎  البحث التقليدي",
        "⬡  البحث المعرفي",
        "◈  التقييم"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("""
<div style='margin-top:20px;padding-top:16px;border-top:0.5px solid rgba(255,255,255,0.1);
     font-size:10px;color:rgba(255,255,255,0.28);text-align:center;line-height:2'>
  GPT-4o mini · ChromaDB · Neo4j<br>
  <span style='color:rgba(255,255,255,0.18)'>Haya Alwizrah · Sarah ALowjan</span>
</div>
""", unsafe_allow_html=True)

# الصفحة الرئيسية
if menu == "🏠  الرئيسية":

    st.markdown("""
    <div class="shurah-header">
      <div>
        <h1>لوحة التحكم</h1>
        <p>مقارنة Standard RAG vs Graph RAG لتحليل المعلقات العربية</p>
      </div>
      <span style="font-size:38px">📜</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
      <div class="stat-card">
        <span class="icon">⬡</span>
        <span class="val">1,289</span>
        <span class="lbl">عقدة في Neo4j</span>
      </div>
      <div class="stat-card">
        <span class="icon">↔</span>
        <span class="val">1,023</span>
        <span class="lbl">علاقة (Relationship)</span>
      </div>
      <div class="stat-card">
        <span class="icon">📜</span>
        <span class="val">3</span>
        <span class="lbl">معلقات مُدرَجة</span>
      </div>
      <div class="stat-card">
        <span class="icon">◎</span>
        <span class="val">250</span>
        <span class="lbl">بيت شعري (Verse)</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(KG_SVG, unsafe_allow_html=True)
    st.markdown(COMPARE_HTML, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="content-card">
          <h3>المعلقات المُدرَجة</h3>
          <p class="sub">3 من أصل 10 معلقات عربية كبرى</p>
          <div class="dev-row">
            <div class="avatar av2">عن</div>
            <div>
              <div class="dev-name">معلقة عنترة بن شداد</div>
              <div class="dev-sub">Dataset/معلقة عنترة بن شداد.txt</div>
            </div>
          </div>
          <div class="dev-row">
            <div class="avatar av1">لب</div>
            <div>
              <div class="dev-name">معلقة لبيد بن ربيعة</div>
              <div class="dev-sub">Dataset/معلقة لبيد بن ربيعة.txt</div>
            </div>
          </div>
          <div class="dev-row">
            <div class="avatar av3">أع</div>
            <div>
              <div class="dev-name">معلقة الأعشى</div>
              <div class="dev-sub">Dataset/معلقة الأعشى.txt</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="content-card">
          <h3>النماذج والتقنيات المستخدمة</h3>
          <p class="sub">Embedding · LLM · Databases</p>
          <div class="tech-row">
            <span class="tech-lbl">Embedding Model</span>
            <span class="tech-val" style="color:#3C3489">Sarah0001/Arabic_embed_model</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">Base model</span>
            <span class="tech-val" style="color:#888;font-size:10.5px">Harrier-Arabic-Matryoshka-270m</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">النوع</span>
            <span class="tech-val" style="color:#555;font-size:10.5px">Distilled Arabic Sentence Transformer</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">LLM</span>
            <span class="tech-val" style="color:#185FA5">GPT-4o mini</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">Vector DB</span>
            <span class="tech-val" style="color:#5B2D0E">ChromaDB</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">Graph DB</span>
            <span class="tech-val" style="color:#0F6E56">Neo4j AuraDB</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">Evaluation</span>
            <span class="tech-val" style="color:#3B6D11">RAGAS Framework</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card">
      <h3>فريق التطوير</h3>
      <p class="sub">المشروع من تطوير</p>
      <div style="display:flex;gap:16px;flex-wrap:wrap">
        <div class="dev-row" style="flex:1;min-width:220px;border-bottom:none">
          <div class="avatar av1" style="width:42px;height:42px;font-size:14px">HA</div>
          <div>
            <div class="dev-name" style="font-size:14px">Haya Alwizrah</div>
            <div class="dev-sub">
              <a href="https://github.com/Haya-Alwizrah"
                 style="color:#534AB7;text-decoration:none">
                github.com/Haya-Alwizrah
              </a>
            </div>
          </div>
        </div>
        <div class="dev-row" style="flex:1;min-width:220px;border-bottom:none">
          <div class="avatar av2" style="width:42px;height:42px;font-size:14px">SA</div>
          <div>
            <div class="dev-name" style="font-size:14px">Sarah ALowjan</div>
            <div class="dev-sub" style="color:#9A8F78">المطوّرة المشاركة</div>
          </div>
        </div>
      </div>
    </div>

    <div class="footer-note">
      مشروع تخرج · نظام RAG متخصص لتحليل المعلقات العربية ·
      يقارن بين البحث المتجهي التقليدي (ChromaDB) والبحث المعرفي بالرسم البياني (Neo4j)
    </div>
    """, unsafe_allow_html=True)


# البحث التقليدي
elif menu == "◎  البحث التقليدي":

    st.markdown("""
    <div class="shurah-header">
      <div>
        <h1>البحث التقليدي</h1>
        <p>Vector Search في ChromaDB باستخدام Sarah0001/Arabic_embed_model</p>
      </div>
      <span class="badge-std">Standard RAG</span>
    </div>

    <div class="content-card">
      <h3>آلية عمل النظام</h3>
      <p class="sub">Standard RAG Pipeline</p>
      <p style="font-size:13px;color:#4A3D30;line-height:2">
        يُحوّل النص إلى vector embedding باستخدام نموذج
        <b>Sarah0001/Arabic_embed_model</b>
        (نسخة مُقطَّرة من Harrier-Arabic-Matryoshka-270m — Sentence Transformer للعربية)،
        ثم يبحث في ChromaDB بالتشابه الدلالي ويُمرّر النتائج إلى
        <b>GPT-4o mini</b> للإجابة.
      </p>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "اكتب سؤالك هنا",
        placeholder="مثال: ما إعراب ودع هريرة؟"
    )

    if query:
        with st.spinner("جاري البحث في ChromaDB..."):
            ans, ctx = std_rag.ask(query)
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.chat_message("assistant").write(ans)
            st.markdown('</div>', unsafe_allow_html=True)
            with st.expander("📚 المصادر المسترجعة"):
                st.write(ctx)


# البحث المعرفي
elif menu == "⬡  البحث المعرفي":

    st.markdown("""
    <div class="shurah-header">
      <div>
        <h1>البحث المعرفي</h1>
        <p>Graph RAG — تحليل العلاقات في Neo4j: Verse · Meaning · Grammar · Vocabulary</p>
      </div>
      <span class="badge-grph">Graph RAG</span>
    </div>

    <div class="content-card">
      <h3>آلية عمل النظام</h3>
      <p class="sub">Graph RAG Pipeline</p>
      <p style="font-size:13px;color:#4A3D30;line-height:2">
        يستعلم من قاعدة Neo4j عبر علاقات <b>HAS_MEANING</b> و<b>HAS_GRAMMAR</b>
        و<b>HAS_VOCABULARY</b> لتجميع السياق المترابط للبيت الشعري،
        ثم يُمرّره إلى <b>GPT-4o mini</b> لتقديم إجابة تحليلية تربط المعنى
        بالإعراب والمفردات في سياق واحد متكامل.
      </p>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "اسأل عن تحليل أو علاقة شعرية",
        placeholder="مثال: اشرح البيت الذي يصف فيه عنترة فرسه وأعربه."
    )

    if query:
        with st.spinner("جاري تحليل العلاقات في Neo4j..."):
            ans, ctx = g_rag.ask(query)
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.chat_message("assistant").write(ans)
            st.markdown('</div>', unsafe_allow_html=True)
            with st.expander("📚 البيانات المسترجعة من النود"):
                st.success(ctx)


# التقييم
elif menu == "◈  التقييم":

    st.markdown("""
    <div class="shurah-header">
      <div>
        <h1>لوحة التقييم</h1>
        <p>مقارنة أداء Standard RAG و Graph RAG بمقاييس RAGAS</p>
      </div>
      <span style="font-size:30px">◈</span>
    </div>

    <div class="content-card">
      <h3>مقاييس RAGAS المستخدمة</h3>
      <p class="sub">أربعة مقاييس لتقييم جودة الاسترجاع والإجابة</p>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
        <div style="padding:10px 14px;background:#F9F6EE;border-radius:8px;border:0.5px solid #E2D9C4">
          <div style="font-size:12px;font-weight:600;color:#1A1208">Faithfulness</div>
          <div style="font-size:11px;color:#6A6050;margin-top:2px">مدى وفاء الإجابة للسياق المسترجع</div>
        </div>
        <div style="padding:10px 14px;background:#F9F6EE;border-radius:8px;border:0.5px solid #E2D9C4">
          <div style="font-size:12px;font-weight:600;color:#1A1208">Answer Relevancy</div>
          <div style="font-size:11px;color:#6A6050;margin-top:2px">مدى صلة الإجابة بالسؤال المطروح</div>
        </div>
        <div style="padding:10px 14px;background:#F9F6EE;border-radius:8px;border:0.5px solid #E2D9C4">
          <div style="font-size:12px;font-weight:600;color:#1A1208">Context Precision</div>
          <div style="font-size:11px;color:#6A6050;margin-top:2px">دقة السياق المسترجع وملاءمته</div>
        </div>
        <div style="padding:10px 14px;background:#F9F6EE;border-radius:8px;border:0.5px solid #E2D9C4">
          <div style="font-size:12px;font-weight:600;color:#1A1208">Context Recall</div>
          <div style="font-size:11px;color:#6A6050;margin-top:2px">استرجاع السياق الكافي للإجابة</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀  بدء التقييم"):
        with st.spinner("جاري حساب النتائج... قد يستغرق بضع دقائق"):
            try:
                res_std_obj   = eval_sys.ragas_eval(std_rag, EVAL_PATH)
                res_graph_obj = eval_sys.ragas_eval(g_rag,   EVAL_PATH)

                metrics = [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall"
                ]

                def get_average(scores):
                    if not scores:
                        return 0
                    return round(sum(scores) / len(scores), 3)

                comparison_data = {
                    "المقياس": [m.replace('_', ' ').title() for m in metrics],
                    "Standard RAG": [get_average(res_std_obj[m]) for m in metrics],
                    "Graph RAG":    [get_average(res_graph_obj[m]) for m in metrics]
                }

                st.session_state.evaluation_results = pd.DataFrame(comparison_data)
                st.success("✅ تم التقييم بنجاح!")

            except Exception as e:
                st.error(f"حدث خطأ: {e}")

    if st.session_state.evaluation_results is not None:
        st.markdown('<div class="content-card"><h3>📈 نتائج المقارنة</h3>', unsafe_allow_html=True)
        st.table(st.session_state.evaluation_results)
        st.bar_chart(st.session_state.evaluation_results.set_index("المقياس"))
        st.markdown('</div>', unsafe_allow_html=True)

        csv = st.session_state.evaluation_results.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "⬇️  تحميل النتائج CSV",
            data=csv,
            file_name='evaluation_results.csv'
        )
    else:
        st.info("اضغط على 'بدء التقييم' لمقارنة أداء النظامين.")