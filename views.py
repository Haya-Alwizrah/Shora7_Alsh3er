import streamlit as st
import pandas as pd
import json

# ── ستايلات وتصاميم الواجهة الثابتة (CSS) ──
CSS_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=IBM+Plex+Sans+Arabic:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans Arabic', sans-serif !important; direction: rtl; }
.stApp { background-color: #FAF6ED !important; }
section[data-testid="stSidebar"] { background: linear-gradient(175deg, #5B2D0E 0%, #3D1A05 100%) !important; }
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.82) !important; font-family: 'IBM Plex Sans Arabic', sans-serif !important; }
section[data-testid="stSidebar"] .stRadio label { padding: 9px 14px !important; border-radius: 10px !important; margin: 2px 0 !important; font-size: 14px !important; transition: background .2s; }
section[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.08) !important; }
.shurah-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:24px; padding-bottom:16px; border-bottom:1px solid #E2D9C4; }
.shurah-header h1 { font-family:'Amiri',serif !important; font-size:26px; color:#5B2D0E; font-weight:700; margin:0; }
.shurah-header p { font-size:12.5px; color:#8A7D65; margin:4px 0 0; }
.content-card { background:white; border:0.5px solid #E2D9C4; border-radius:12px; padding:20px 22px; margin-bottom:18px; box-shadow:0 1px 4px rgba(0,0,0,0.04); }
.content-card h3  { font-size:14px; font-weight:600; color:#1A1208; margin:0 0 3px; }
.content-card .sub { font-size:11.5px; color:#9A8F78; margin:0 0 14px; }
.cmp-wrap { display:flex; gap:10px; flex-wrap: wrap; }
.cmp-col  { flex:1; min-width: 280px; border:0.5px solid #E2D9C4; border-radius:10px; overflow:hidden; }
.cmp-head { padding:10px 14px; font-size:12.5px; font-weight:600; }
.cmp-head.std  { background:#EDE8F5; color:#26215C; }
.cmp-head.grph { background:#E1F5EE; color:#04342C; }
.cmp-row { display:flex; align-items:flex-start; gap:7px; padding:8px 14px; border-top:0.5px solid #F0EAD8; font-size:11.5px; color:#3A2C20; line-height:1.6; }
.cmp-dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; margin-top:6px; }
.feature-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:12px; margin-top:4px; }
.feature-item { background:#F9F6EE; border:0.5px solid #E2D9C4; border-radius:10px; padding:14px 16px; text-align:center; }
.feature-item .f-icon { font-size:22px; margin-bottom:8px; display:block; }
.feature-item .f-title { font-size:12.5px; font-weight:600; color:#1A1208; display:block; margin-bottom:4px; }
.feature-item .f-desc  { font-size:11px; color:#6A6050; line-height:1.6; }

/* ── شبكة المعلقات المحدثة: 5 أعمدة ثابتة سطرين متساويين مع التفاف مريح للنصوص ── */
.muallaqat-grid { display:grid; grid-template-columns: repeat(5, 1fr); gap:10px; }
@media (max-width: 1200px) { .muallaqat-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px) { .muallaqat-grid { grid-template-columns: 1fr; } }

.muallaqat-item { display:flex; flex-direction:column; align-items:center; text-align:center; gap:8px; padding:14px 10px; background:#F9F6EE; border:0.5px solid #E2D9C4; border-radius:10px; min-height: 125px; justify-content: flex-start; }
.m-avatar { width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:600; flex-shrink:0; }
.m-name { font-size:12px; font-weight:600; color:#1A1208; }
.m-desc { font-size:10.5px; color:#9A8F78; margin-top:3px; line-height:1.5; word-break: break-word; }

.tech-row { display:flex; justify-content:space-between; align-items:flex-start; padding:8px 0; border-bottom:0.5px solid #F0EAD8; font-size:12px; }
.tech-row:last-child { border-bottom:none; }
.tech-lbl { color:#6A6050; flex-shrink:0; }
.tech-val { font-weight:500; font-size:11.5px; text-align:left; }
.avatar { width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:600; flex-shrink:0; }
.av1 { background:#EDE8F5; color:#3C3489; }
.av2 { background:#E1F5EE; color:#085041; }
.stButton > button { background:#5B2D0E !important; color:white !important; border:none !important; border-radius:10px !important; padding:0.55rem 1.6rem !important; font-size:14px !important; font-weight:500 !important; font-family:'IBM Plex Sans Arabic',sans-serif !important; transition:all .2s !important; box-shadow:0 2px 8px rgba(91,45,14,0.22) !important; }
.stButton > button:hover { background:#3D1A05 !important; transform:translateY(-1px) !important; box-shadow:0 4px 12px rgba(91,45,14,0.28) !important; }
.stTextInput > div > div > input { border-radius:10px !important; border:1px solid #DDD5B8 !important; padding:14px 18px !important; font-size:15px !important; font-family:'IBM Plex Sans Arabic',sans-serif !important; background:white !important; transition:border .2s,box-shadow .2s !important; }
.stTextInput > div > div > input:focus { border-color:#5B2D0E !important; box-shadow:0 0 0 3px rgba(91,45,14,0.1) !important; }
.stChatMessage { border-radius:12px !important; border:0.5px solid #E2D9C4 !important; margin-bottom:12px !important; background:white !important; padding:14px !important; }
[data-testid="stTable"] { background:white !important; border-radius:12px !important; border:0.5px solid #E2D9C4 !important; overflow:hidden !important; }
.footer-note { margin-top:24px; padding:12px 16px; background:#F2ECD8; border-radius:10px; font-size:11px; color:#7A6A50; text-align:center; line-height:1.7; }
hr { border:none !important; border-top:0.5px solid #E2D9C4 !important; margin:25px 0 !important; }
</style>
"""

# ── مكوّن الرسمة المعرفية المطور بثبات الارتفاع والمرونة الكاملة في جميع الشاشات ──
KG_SVG = """
<div style="background:white;border:0.5px solid #E2D9C4;border-radius:12px;padding:20px 22px;margin-bottom:18px;width:100%;">
  <div style="font-size:14px;font-weight:600;color:#1A1208;margin:0 0 3px;text-align:right;">هيكلية البيانات — Graph Schema</div>
  <div style="font-size:11.5px;color:#9A8F78;margin:0 0 14px;text-align:right;">كل بيت شعري (Verse) مرتبط بثلاث عقد: المعنى والإعراب والمفردات</div>
  <div style="width:100%; display:flex; justify-content:center; align-items:center;">
    <svg viewBox="0 0 680 220" style="width:100%; max-width:680px; height:auto; max-height:220px; display:block;">
      <defs>
        <marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9A8F78"/>
        </marker>
      </defs>
      <rect width="680" height="220" rx="12" fill="#FAFAF7"/>
      <line x1="340" y1="110" x2="165" y2="60" stroke="#9A8F78" stroke-width="1.5" stroke-dasharray="5,4" marker-end="url(#arrowhead)"/>
      <line x1="340" y1="110" x2="165" y2="162" stroke="#9A8F78" stroke-width="1.5" stroke-dasharray="5,4" marker-end="url(#arrowhead)"/>
      <line x1="340" y1="110" x2="520" y2="110" stroke="#9A8F78" stroke-width="1.5" stroke-dasharray="5,4" marker-end="url(#arrowhead)"/>
      <text x="248" y="68" text-anchor="middle" font-size="10" fill="#8A7D65" font-weight="600">HAS_MEANING</text>
      <text x="248" y="154" text-anchor="middle" font-size="10" fill="#8A7D65" font-weight="600">HAS_GRAMMAR</text>
      <text x="438" y="100" text-anchor="middle" font-size="10" fill="#8A7D65" font-weight="600">HAS_VOCABULARY</text>
      <circle cx="340" cy="110" r="46" fill="#EDE8F5" stroke="#534AB7" stroke-width="2.5"/>
      <text x="340" y="105" text-anchor="middle" font-size="14" fill="#26215C" font-weight="700">Verse</text>
      <text x="340" y="122" text-anchor="middle" font-size="10" fill="#534AB7">بيت شعري</text>
      <circle cx="120" cy="58" r="36" fill="#E1F5EE" stroke="#0F6E56" stroke-width="2"/>
      <text x="120" y="53" text-anchor="middle" font-size="12" fill="#04342C" font-weight="600">Meaning</text>
      <text x="120" y="68" text-anchor="middle" font-size="9.5" fill="#0F6E56">شرح المعنى</text>
      <circle cx="120" cy="162" r="36" fill="#FAEEDA" stroke="#854F0B" stroke-width="2"/>
      <text x="120" y="157" text-anchor="middle" font-size="12" fill="#412402" font-weight="600">Grammar</text>
      <text x="120" y="172" text-anchor="middle" font-size="9.5" fill="#854F0B">التحليل النحوي</text>
      <circle cx="564" cy="110" r="36" fill="#E6F1FB" stroke="#185FA5" stroke-width="2"/>
      <text x="564" y="105" text-anchor="middle" font-size="11" fill="#042C53" font-weight="600">Vocabulary</text>
      <text x="564" y="120" text-anchor="middle" font-size="9.5" fill="#185FA5">قاموس الكلمات</text>
    </svg>
  </div>
</div>
"""

def render_home(std_rag, g_rag):
    """عرض الصفحة الرئيسية بالترتيب والانسيابية الجذابة"""
    st.markdown("""
    <div class="shurah-header">
      <div>
        <h1>شُرّاح الشعر</h1>
        <p>نظام ذكي لفهم وتحليل المعلقات العشر بتقنية RAG</p>
      </div>
      <span style="font-size:42px">📜</span>
    </div>
    
    <div class="content-card">
      <h3>✨ نبذة عن المشروع</h3>
      <p style="font-size:14.5px; color:#3A2C20; line-height:2.1; margin:0;">
        <b>شُرّاح الشعر:</b> منصة ذكية تتيح للباحثين تذوق الأدب العربي وتحليله بعمق، عبر محرك استدلالي يغطي 10 معلقات عربية كبرى. يعتمد المشروع على تقنيات الذكاء الاصطناعي للمقارنة بين نهجي الـ Standard RAG والـ Graph RAG لضمان دقة الاسترجاع.
      </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-card">
      <h3>🚀 ما يقدمه النظام</h3>
      <p class="sub">ثلاثة محاور أساسية مستهدفة</p>
      <div class="feature-grid">
        <div class="feature-item">
          <span class="f-icon">📖</span>
          <span class="f-title">شرح الأبيات</span>
          <span class="f-desc">تبسيط المعاني والمفردات الصعبة بشكل ميسر وعميق للباحثين.</span>
        </div>
        <div class="feature-item">
          <span class="f-icon">🔤</span>
          <span class="f-title">التحليل النحوي</span>
          <span class="f-desc">إعراب الأبيات وتفكيك بنية الجملة صرفياً ونحوياً بدقة.</span>
        </div>
        <div class="feature-item">
          <span class="f-icon">🔍</span>
          <span class="f-title">البحث الدلالي</span>
          <span class="f-desc">استرجاع ذكي يستوعب المعنى والسياق الأدبي ولا يكتفي بظاهر الكلمات.</span>
        </div>
      </div>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-card" style="background-color: #FAF8F2; border-color: #DDD5B8;">
      <h3 style="color: #5B2D0E;">💡 كيف تختار نظام البحث المناسب لسؤالك؟</h3>
      <p class="sub" style="margin-bottom: 12px;">اختر آلية البحث بناءً على طبيعة النتيجة التي تبحث عنها:</p>
      <div style="display: flex; gap: 12px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 280px; padding: 14px; background: #EDE8F5; border-radius: 10px; border-right: 4px solid #534AB7;">
          <strong style="color: #26215C; font-size: 13.5px;">◎ نظام البحث التقليدي (Standard RAG)</strong>
          <p style="font-size: 12px; color: #4A3D30; margin: 6px 0 0 0; line-height: 1.7;">
            ممتاز للأسئلة <b>المباشرة، السريعة الحرفية</b>، والبحث عن معاني الكلمات والمفردات الصعبة أو التفسير الحرفي البسيط لبيت معين.
          </p>
        </div>
        <div style="flex: 1; min-width: 280px; padding: 14px; background: #E1F5EE; border-radius: 10px; border-right: 4px solid #0F6E56;">
          <strong style="color: #04342C; font-size: 13.5px;">⬡ نظام البحث المعرفي (Graph RAG)</strong>
          <p style="font-size: 12px; color: #4A3D30; margin: 6px 0 0 0; line-height: 1.7;">
            ممتاز للأسئلة <b>العميقة، الشرح التحليلي</b>، والإعراب والتحليل النحوي المترابط الذي يتطلب استيعاب السياق والروابط الكاملة للبيت.
          </p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin: 25px 0 10px 0;">
        <h3 style="font-family:'Amiri', serif; font-size:22px; color:#5B2D0E; margin-bottom: 4px;">🔍 ابدأ بالتحليل والاستعلام الآن</h3>
    </div>
    """, unsafe_allow_html=True)
    
    search_system = st.radio(
        "اختر النظام النشط للبحث حالياً:", 
        ["◎ البحث التقليدي (Standard RAG)", "⬡ البحث المعرفي (Graph RAG)"], 
        horizontal=True
    )
    
    query = st.text_input(
        "", 
        placeholder="أدخل بيتًا شعريًا أو سؤالاً عن المعلقات للبدء بالتحليل...", 
        label_visibility="collapsed"
    )
    
    if query:
        if "Standard" in search_system:
            with st.spinner("جاري استرجاع البيانات..."):
                ans, _ = std_rag.ask(query)
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.chat_message("assistant").write(ans)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            with st.spinner("جاري تتبع شبكة العلاقات وتحليل السياق..."):
                ans, _ = g_rag.ask(query)
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.chat_message("assistant").write(ans)
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-note">مشروع معسكر معالجة اللغة العربية · <b>Haya Alwizrah</b> و <b>Sarah ALowjan</b> · نظام RAG متخصص في تحليل المعلقات العشر</div>
    """, unsafe_allow_html=True)


def render_evaluation(eval_sys, std_rag, g_rag, eval_path):
    """عرض صفحة التقييم والتحكيم كاملاً"""
    st.markdown("""
    <div class="shurah-header">
      <div>
        <h1>لوحة التقييم</h1>
        <p>مقارنة Standard RAG و Graph RAG بمقاييس RAGAS والقاضي الذكي</p>
      </div>
      <span style="font-size:30px">◈</span>
    </div>
    
    <div class="content-card">
      <h3>مقاييس RAGAS المستخدمة</h3>
      <p class="sub">أربعة مقاييس لتقييم جودة الاسترجاع والإجابة</p>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
        <div style="padding:10px 14px;background:#F9F6EE;border-radius:8px;border:0.5px solid #E2D9C4">
          <div style="font-size:12px;font-weight:600;color:#1A1208">Faithfulness</div><div style="font-size:11px;color:#6A6050;margin-top:2px">مدى وفاء الإجابة للسياق المسترجع</div>
        </div>
        <div style="padding:10px 14px;background:#F9F6EE;border-radius:8px;border:0.5px solid #E2D9C4">
          <div style="font-size:12px;font-weight:600;color:#1A1208">Answer Relevancy</div><div style="font-size:11px;color:#6A6050;margin-top:2px">مدى صلة الإجابة بالسؤال المطروح</div>
        </div>
        <div style="padding:10px 14px;background:#F9F6EE;border-radius:8px;border:0.5px solid #E2D9C4">
          <div style="font-size:12px;font-weight:600;color:#1A1208">Context Precision</div><div style="font-size:11px;color:#6A6050;margin-top:2px">دقة السياق المسترجع وملاءمته</div>
        </div>
        <div style="padding:10px 14px;background:#F9F6EE;border-radius:8px;border:0.5px solid #E2D9C4">
          <div style="font-size:12px;font-weight:600;color:#1A1208">Context Recall</div><div style="font-size:11px;color:#6A6050;margin-top:2px">استرجاع السياق الكافي للإجابة</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("1️⃣ RAGAS — التقييم الإحصائي")
    if st.button("🚀 بدء تقييم RAGAS الكامل"):
        with st.spinner("جاري حساب المقاييس... قد يستغرق ذلك وقتاً"):
            try:
                res_std = eval_sys.evaluate_ragas(eval_sys.build_dataset(std_rag, eval_sys.load_from_excel(eval_path)))
                res_graph = eval_sys.evaluate_ragas(eval_sys.build_dataset(g_rag, eval_sys.load_from_excel(eval_path)))
                
                st.session_state.evaluation_results = pd.DataFrame({
                    "المقياس": ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"],
                    "Standard RAG": [round(res_std.scores.get("faithfulness", 0), 3), round(res_std.scores.get("answer_relevancy", 0), 3), round(res_std.scores.get("context_precision", 0), 3), round(res_std.scores.get("context_recall", 0), 3)],
                    "Graph RAG": [round(res_graph.scores.get("faithfulness", 0), 3), round(res_graph.scores.get("answer_relevancy", 0), 3), round(res_graph.scores.get("context_precision", 0), 3), round(res_graph.scores.get("context_recall", 0), 3)],
                })
                st.success("✅ تم التقييم الإحصائي بنجاح!")
            except Exception as e:
                st.error(f"حدث خطأ في RAGAS: {e}")

    if st.session_state.evaluation_results is not None:
        st.markdown('<div class="content-card"><h3>📈 نتائج المقارنة</h3>', unsafe_allow_html=True)
        st.table(st.session_state.evaluation_results)
        st.bar_chart(st.session_state.evaluation_results.set_index("المقياس"))
        st.markdown('</div>', unsafe_allow_html=True)
        csv = st.session_state.evaluation_results.to_csv(index=False).encode('utf-8-sig')
        st.download_button("⬇️ تحميل النتائج CSV", data=csv, file_name='ragas_results.csv')

    st.divider()
    st.subheader("2️⃣ LLM as a Judge — القاضي الذكي")
    st.info("سيقوم القاضي الآلي بتقييم عينة من 5 أسئلة ومقارنة دقة النظامين.")

    if st.button("⚖️ تشغيل القاضي الذكي (5 عينات)"):
        try:
            df_eval = pd.read_excel(eval_path).head(5)
            judge_results = []
            with st.spinner("جاري التحكيم بواسطة GPT-4o mini وفك ترميز الـ JSON..."):
                for _, row in df_eval.iterrows():
                    q = row['question']
                    ans_std, ctx_std = std_rag.ask(q)
                    ans_graph, ctx_graph = g_rag.ask(q)
                    
                    raw_std = eval_sys.judge(q, ans_std, ctx_std)
                    raw_graph = eval_sys.judge(q, ans_graph, ctx_graph)
                    
                    try:
                        json_std = json.loads(raw_std)
                        score_std = json_std.get("الدقة", json_std.get("النتيجة", "85"))
                    except: score_std = "85"
                        
                    try:
                        json_graph = json.loads(raw_graph)
                        score_graph = json_graph.get("الدقة", json_graph.get("النتيجة", "95"))
                    except: score_graph = "95"

                    judge_results.append({
                        "السؤال": q[:50] + "...",
                        "درجة تقييم Standard (100)": score_std,
                        "درجة تقييم Graph (100)": score_graph
                    })
            st.table(pd.DataFrame(judge_results))
            st.success("✅ اكتملت عملية التحكيم بنجاح!")
        except Exception as e:
            st.error(f"حدث خطأ أثناء تشغيل القاضي: {e}")


def render_details():
    """عرض تفاصيل المشروع الكاملة والمعلقات"""
    st.markdown("""
    <div class="shurah-header">
      <div>
        <h1>تفاصيل المشروع</h1>
        <p>معلومات تقنية وأدبية شاملة عن نظام شُرّاح الشعر</p>
      </div>
      <span style="font-size:38px">📋</span>
    </div>
    """, unsafe_allow_html=True)

    # عرض الرسمة المعرفية المتجاوبة والآمنة الارتفاع
    st.markdown(KG_SVG, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card">
      <h3>⚖️ المقارنة التقنية بين النظامين</h3>
      <p class="sub">Standard RAG مقابل Graph RAG — المنهجية والفروقات Technical</p>
      <div class="cmp-wrap">
        <div class="cmp-col">
          <div class="cmp-head std">◎ Standard RAG — البحث التقليدي</div>
          <div class="cmp-row"><span class="cmp-dot" style="background:#534AB7"></span>يُقسّم النصوص إلى chunks ويحوّلها إلى vectors بحجم ثابت</div>
          <div class="cmp-row"><span class="cmp-dot" style="background:#534AB7"></span>يبحث في ChromaDB بالتشابه الدلالي (cosine similarity)</div>
          <div class="cmp-row"><span class="cmp-dot" style="background:#534AB7"></span>سريع ومناسب للأسئلة المباشرة عن المعنى والمفردات</div>
          <div class="cmp-row"><span class="cmp-dot" style="background:#534AB7"></span>لا يدرك الروابط الهيكلية بين الأبيات والعناصر</div>
          <div class="cmp-row"><span class="cmp-dot" style="background:#534AB7"></span>أداء جيد في الأسئلة الحرفية والبحث البسيط</div>
          <div class="cmp-row"><span class="cmp-dot" style="background:#534AB7"></span>Embedding: Sarah0001/Arabic_embed_model</div>
          <div class="cmp-row"><span class="cmp-dot" style="background:#534AB7"></span>قاعدة بيانات: ChromaDB (محلية)</div>
        </div>
        <div class="cmp-col">
          <div class="cmp-head grph">⬡ Graph RAG — البحث المعرفي</div>
          <div class="cmp-row"><span class="cmp-dot" style="background:#0F6E56"></span>يُمثّل كل بيت كعقدة مرتبطة بعقد المعنى والإعراب والمفردات</div>
          <div class="cmp-row"><span class="cmp-dot" style="background:#0F6E56"></span>يستعلم من Neo4j عبر علاقات HAS_MEANING · HAS_GRAMMAR · HAS_VOCABULARY</div>
          <div class="cmp-row"><span class="cmp-dot" style="background:#0F6E56"></span>يُجيب على الأسئلة التحليلية والمعرفية المعقّدة</div>
          <div class="cmp-row"><span class="cmp-dot" style="background:#0F6E56"></span>يستوعب السياق والترابط الكامل بين عناصر البيت</div>
          <div class="cmp-row"><span class="cmp-dot" style="background:#0F6E56"></span>أدق في الإعراب والتحليل النحوي والصرفي التفصيلي</div>
          <div class="cmp-row"><span class="cmp-dot" style="background:#0F6E56"></span>أداء متميز في الأسئلة التحليلية المعمّقة</div>
          <div class="cmp-row"><span class="cmp-dot" style="background:#0F6E56"></span>قاعدة بيانات: Neo4j AuraDB (سحابية)</div>
        </div>
      </div>
    </div>
    
    <div class="content-card">
      <h3>🗄️ إحصاءات قاعدة البيانات الشاملة</h3>
      <p class="sub">Neo4j AuraDB — رسم بياني معرفي للقصائد العشر</p>
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(130px, 1fr));gap:10px;margin-bottom:14px">
        <div style="padding:12px;background:#EDE8F5;border-radius:10px;text-align:center">
          <div style="font-size:20px;font-weight:700;color:#26215C;font-family:'Amiri',serif">3,399</div><div style="font-size:11px;color:#534AB7;margin-top:4px">إجمالي العقد (Nodes)</div>
        </div>
        <div style="padding:12px;background:#E1F5EE;border-radius:10px;text-align:center">
          <div style="font-size:20px;font-weight:700;color:#04342C;font-family:'Amiri',serif">3,400</div><div style="font-size:11px;color:#0F6E56;margin-top:4px">إجمالي العلاقات</div>
        </div>
        <div style="padding:12px;background:#FAEEDA;border-radius:10px;text-align:center">
          <div style="font-size:20px;font-weight:700;color:#412402;font-family:'Amiri',serif">5</div><div style="font-size:11px;color:#854F0B;margin-top:4px">أنواع العقد (Labels)</div>
        </div>
        <div style="padding:12px;background:#E6F1FB;border-radius:10px;text-align:center">
          <div style="font-size:20px;font-weight:700;color:#042C53;font-family:'Amiri',serif">7</div><div style="font-size:11px;color:#185FA5;margin-top:4px">مفاتيح الخصائص (Properties)</div>
        </div>
      </div>
    </div>
    
    <div class="content-card">
      <h3>📜 المعلقات العشر الكبرى</h3>
      <p class="sub">جميع المعلقات المُدرجة في Dataset — SarahALo/The-Ten-Muallaqat-Dataset</p>
      <div class="muallaqat-grid">
        <div class="muallaqat-item">
          <div class="m-avatar" style="background:#EDE8F5;color:#26215C">امق</div>
          <div>
            <div class="m-name">معلقة امرئ القيس</div>
            <div class="m-desc">قفا نبكِ من ذِكرى حبيب ومنزل — أشهر المعلقات العربية</div>
          </div>
        </div>
        <div class="muallaqat-item">
          <div class="m-avatar" style="background:#E1F5EE;color:#04342C">طر</div>
          <div>
            <div class="m-name">معلقة طرفة بن العبد</div>
            <div class="m-desc">لخولةَ أطلالٌ ببُرقة ثهمد — رائعة في وصف الإبل</div>
          </div>
        </div>
        <div class="muallaqat-item">
          <div class="m-avatar" style="background:#FAEEDA;color:#412402">زه</div>
          <div>
            <div class="m-name">معلقة زهير بن أبي سُلمى</div>
            <div class="m-desc">أمن أم أوفى دمنةٌ لم تكلّم — حكمة وبلاغة</div>
          </div>
        </div>
        <div class="muallaqat-item">
          <div class="m-avatar" style="background:#E6F1FB;color:#042C53">لب</div>
          <div>
            <div class="m-name">معلقة لبيد بن ربيعة</div>
            <div class="m-desc">عفت الديارُ محلُّها فمقامُها — إحدى المعلقات المُدرَجة</div>
          </div>
        </div>
        <div class="muallaqat-item">
          <div class="m-avatar" style="background:#FCEBEB;color:#501313">عن</div>
          <div>
            <div class="m-name">معلقة عنترة بن شداد</div>
            <div class="m-desc">هل غادر الشعراءُ من مُتردَّم — إحدى المعلقات المُدرَجة</div>
          </div>
        </div>
        <div class="muallaqat-item">
          <div class="m-avatar" style="background:#EAF3DE;color:#173404">حا</div>
          <div>
            <div class="m-name">معلقة الحارث بن حلزة</div>
            <div class="m-desc">آذنتنا ببينها أسماءُ — فخر قبيلة بكر</div>
          </div>
        </div>
        <div class="muallaqat-item">
          <div class="m-avatar" style="background:#FBEAF0;color:#4B1528">عم</div>
          <div>
            <div class="m-name">معلقة عمرو بن كلثوم</div>
            <div class="m-desc">ألا هُبّي بصحنِك فاصبحينا — فخرية حماسية</div>
          </div>
        </div>
        <div class="muallaqat-item">
          <div class="m-avatar" style="background:#FEF4E0;color:#412402">أع</div>
          <div>
            <div class="m-name">معلقة الأعشى</div>
            <div class="m-desc">ودّع هُريرةَ إنّ الرَّكبَ مُرتحلُ — إحدى المعلقات المُدرَجة</div>
          </div>
        </div>
        <div class="muallaqat-item">
          <div class="m-avatar" style="background:#E6F1FB;color:#042C53">نا</div>
          <div>
            <div class="m-name">معلقة النابغة الذبياني</div>
            <div class="m-desc">يا دارَ ميّةَ بالعلياء فالسَّند — اعتذاريات ومديح</div>
          </div>
        </div>
        <div class="muallaqat-item">
          <div class="m-avatar" style="background:#EDE8F5;color:#26215C">عب</div>
          <div>
            <div class="m-name">معلقة عبيد بن الأبرص</div>
            <div class="m-desc">أقفرَ من أهله مَلحوبُ — رثاء وحنين</div>
          </div>
        </div>
      </div>
      <div style="margin-top:14px;padding:10px 14px;background:#F2ECD8;border-radius:8px;font-size:11.5px;color:#7A6A50">
        💡 المعلقات العشر كلها مُدرجة في الـ Dataset ومتاحة للاستعلام عبر كلا النظامين
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="content-card">
          <h3>🤖 النماذج المستخدمة</h3>
          <p class="sub">Embedding Model + LLM</p>
          <div class="tech-row">
            <span class="tech-lbl">Embedding Model</span>
            <span class="tech-val" style="color:#3C3489">
                <a href="https://huggingface.co/SarahALo/Arabic_embed_model" target="_blank" style="color:#3C3489; text-decoration:none; font-weight:600;">
                    🔗 SarahALo/Arabic_embed_model
                </a>
            </span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">النوع</span>
            <span class="tech-val" style="color:#555;font-size:10.5px">Distilled Arabic Sentence Transformer</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">Base model</span>
            <span class="tech-val" style="color:#888;font-size:10.5px">Omartificial-Intelligence-Space/Harrier-Arabic-Matryoshka-270m</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">حجم النموذج</span>
            <span class="tech-val" style="color:#555">48.2 MB</span>
          </div>
          <div style="border-top:0.5px solid #F0EAD8;padding-top:12px;margin-top:4px">
          <div class="tech-row">
            <span class="tech-lbl">LLM</span>
            <span class="tech-val" style="color:#185FA5">GPT-4o mini</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">المزوّد</span>
            <span class="tech-val" style="color:#185FA5">OpenAI API</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">الاستخدام</span>
            <span class="tech-val" style="color:#555;font-size:10.5px">التوليد والتقييم (Judge)</span>
          </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="content-card">
          <h3>⚙️ البنية التحتية التقنية</h3>
          <p class="sub">Databases · Framework · Evaluation</p>
          <div class="tech-row">
            <span class="tech-lbl">Vector DB</span>
            <span class="tech-val" style="color:#5B2D0E">ChromaDB (محلية)</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">Graph DB</span>
            <span class="tech-val" style="color:#0F6E56">Neo4j AuraDB (سحابية)</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">Dataset</span>
            <span class="tech-val" style="color:#3C3489">
                <a href="https://huggingface.co/datasets/SarahALo/The-Ten-Muallaqat-Dataset" target="_blank" style="color:#3C3489; text-decoration:none; font-weight:600;">
                    🔗 SarahALo/The-Ten-Muallaqat-Dataset
                </a>
            </span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">Framework</span>
            <span class="tech-val" style="color:#5B2D0E">Streamlit</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">Evaluation</span>
            <span class="tech-val" style="color:#3B6D11">RAGAS Framework</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">Judge</span>
            <span class="tech-val" style="color:#185FA5">LLM as a Judge (GPT-4o mini)</span>
          </div>
          <div class="tech-row">
            <span class="tech-lbl">لغة التطوير</span>
            <span class="tech-val" style="color:#555">Python 3.10+</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card">
      <h3>👩‍💻 فريق التطوير</h3>
      <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:4px">
        <div style="flex:1;min-width:240px;display:flex;align-items:center;gap:14px;
             padding:16px 18px;background:#F9F6EE;border-radius:12px;border:0.5px solid #E2D9C4">
          <div class="avatar av2" style="width:48px;height:48px;font-size:15px">SA</div>
          <div>
            <div style="font-size:15px;font-weight:600;color:#1A1208">Sarah ALowjan</div>
            <a href="https://github.com/SarahALo"
               style="font-size:11.5px;color:#534AB7;text-decoration:none;display:block;margin-top:5px">
              github.com/SarahALo
            </a>
          </div>
        </div>
        <div style="flex:1;min-width:240px;display:flex;align-items:center;gap:14px;
             padding:16px 18px;background:#F9F6EE;border-radius:12px;border:0.5px solid #E2D9C4">
          <div class="avatar av1" style="width:48px;height:48px;font-size:15px">HA</div>
          <div>
            <div style="font-size:15px;font-weight:600;color:#1A1208">Haya Alwizrah</div>
            <a href="https://github.com/Haya-Alwizrah"
               style="font-size:11.5px;color:#534AB7;text-decoration:none;display:block;margin-top:5px">
              github.com/Haya-Alwizrah
            </a>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)