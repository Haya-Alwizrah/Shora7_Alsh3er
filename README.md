# 📜 Shura7 Al-Sh3er (Arabic Poetry Expert System)

An advanced expert system specialized in analyzing and parsing the **Arabic Mu'allaqat** using dual RAG architectures: **Standard RAG (Vector-based)** and **GraphRAG (Relationship-based)**.

## 👥 Developers
* **Haya Alwizrah** (Haya-Alwizrah)
* **Sarah ALowjan** (Sarah ALowjan)

## 🏗️ Project Architecture
The system follows a strict data lifecycle from raw document extraction to final AI evaluation.

```mermaid
graph TD
    A[Source: فتح الكبير المتعال(اعراب المعلقات العشر) PDF/Images] --> B[Phase 1: OCR Extraction - MistralAI/Qara]
    B --> C[Text Splitting & Cleaning - Regex]
    C --> D[Phase 2: Embedding - Arabic Matryoshka Model]
    D --> E1[Standard RAG Path: Vector Storage - ChromaDB]
    D --> E2[GraphRAG Path: Relationship Building - Neo4j]
    E1 --> F[Phase 3: Retrieval & Generation - GPT-4o Mini]
    E2 --> F
    F --> G[Phase 4: Evaluation - RAGAS Framework]

```

## 🛠️ Technical Stack

* **LLM:** OpenAI `gpt-4o-mini` for high-accuracy reasoning.
* **Embeddings:** `Sarah0001/Arabic_embed_model` for specialized Arabic semantic representation.
* **Databases:**
* **ChromaDB:** Vector store for semantic search.
* **Neo4j AuraDB:** Knowledge graph for structural parsing (Verse → Grammar → Meaning).


* **Frameworks:** Streamlit, LangChain, Ragas.

## 📚 Dataset

Currently, the system covers a curated dataset of 3 major pre-Islamic Mu'allaqat:

1. **Al-A'sha**
2. **'Antara ibn Shaddad**
3. **Labid ibn Rabi'ah**

## 📊 Evaluation Metrics

### 1. RAGAS Framework (Statistical Metrics)

* **Faithfulness:** Detects hallucinations by checking if the answer is grounded in the retrieved verse.
* **Answer Relevancy:** Measures how well the response addresses the specific linguistic or historical query.
* **Context Precision/Recall:** Evaluates the efficiency of both Neo4j and ChromaDB in retrieving relevant parsing data.

### 2. LLM as a Judge (Qualitative Analysis)

* **Expert Peer Review:** A secondary `gpt-4o-mini` instance acts as a "Linguistic Judge".
* **Scoring (1-100):** Evaluates responses based on **Accuracy**, **Context Adherence**, and **Linguistic Nuance**.
* **Comparison:** Provides a direct percentage-based comparison between Standard RAG and GraphRAG performance on complex samples.
## 🚀 Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Configure `.env` with your OpenAI and Neo4j credentials.
3. Run the dashboard: `streamlit run app.py`

---
