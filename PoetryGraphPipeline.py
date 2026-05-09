import re
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


class PoetryPipeline:

    def __init__(self, files):

        # =========================================
        # LOAD ENV
        # =========================================
        load_dotenv()

        self.files = files

        # =========================================
        # OPENAI
        # =========================================
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.model_name = "gpt-4o-mini"

        # =========================================
        # NEO4J
        # =========================================
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(
                os.getenv("NEO4J_USERNAME"),
                os.getenv("NEO4J_PASSWORD")
            )
        )

        # =========================================
        # EMBEDDING MODEL
        # =========================================
        self.embedding_model = SentenceTransformer(
            "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
        )

        # =========================================
        # CACHE
        # =========================================
        self.cache_file = "cache.json"

        if os.path.exists(self.cache_file):

            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

        else:
            self.cache = {}

        # =========================================
        # PROMPT
        # =========================================
        self.prompt_template = """
أنت محلل متخصص في الشعر العربي.

سيصلك نص يحتوي على:

1- بيت شعري واحد فقط
2- قسم المفردات
3- قسم المعنى
4- قسم الإعراب

المطلوب:

استخرج البيانات التالية فقط:

- verse:
النص الشعري فقط بدون الشرح

- vocabulary:
قسم المفردات فقط

- meaning:
قسم المعنى فقط

- grammar:
قسم الإعراب فقط

مهم جداً:

- لا تستخرج أكثر من بيت واحد
- لا تعتبر الشرح أو الإعراب بيتاً شعرياً
- لا تقسم النص
- أعد النتيجة JSON فقط

الصيغة المطلوبة:

[
  {
    "verse": "...",
    "vocabulary": "...",
    "meaning": "...",
    "grammar": "..."
  }
]

النص:
"""

    # =====================================================
    # LOAD DATA
    # =====================================================

    def load_data(self):

        chunks = []

        for f in self.files:

            try:

                with open(f, "r", encoding="utf-8") as file:
                    content = file.read()

                # تقسيم النص
                parts = re.split(r'-{15,}', content)

                for part in parts:

                    chunk = part.strip()

                    if len(chunk) > 50:
                        chunks.append(chunk)

            except Exception as e:

                print(f"❌ Error reading {f}: {e}")

        print(f"✅ Total chunks loaded: {len(chunks)}")

        return chunks

    # =====================================================
    # GPT PARSER
    # =====================================================

    def parse_text(self, text):

        # =========================================
        # CACHE
        # =========================================
        if text in self.cache:

            print("✅ Loaded from cache")

            return self.cache[text]

        # =========================================
        # API CALL
        # =========================================
        response = self.client.chat.completions.create(

            model=self.model_name,

            temperature=0.2,

            messages=[
                {
                    "role": "user",
                    "content": self.prompt_template + text
                }
            ]
        )

        content = response.choices[0].message.content

        try:

            clean_json = re.sub(
                r'```json|```',
                '',
                content
            ).strip()

            result = json.loads(clean_json)

            # =========================================
            # SAVE CACHE
            # =========================================
            self.cache[text] = result

            self.save_cache()

            return result

        except Exception as e:

            print(f"❌ JSON/API Error: {e}")

            return []

    # =====================================================
    # SAVE CACHE
    # =====================================================

    def save_cache(self):

        with open(self.cache_file, "w", encoding="utf-8") as f:

            json.dump(
                self.cache,
                f,
                ensure_ascii=False,
                indent=2
            )

        print("✅ cache saved")

    # =====================================================
    # GENERATE EMBEDDING
    # =====================================================

    def generate_embedding(self, text):

        return self.embedding_model.encode(text).tolist()

    # =====================================================
    # INSERT INTO GRAPH
    # =====================================================

    def insert_graph(self, item):

        with self.driver.session() as session:

            session.execute_write(
                self._create_nodes,
                item
            )

    @staticmethod
    def _create_nodes(tx, item):

        tx.run("""

        MERGE (v:Verse {
            text: $verse
        })

        MERGE (m:Meaning {
            text: $meaning
        })

        MERGE (g:Grammar {
            text: $grammar
        })

        MERGE (vo:Vocabulary {
            text: $vocab
        })

        MERGE (v)-[:HAS_MEANING]->(m)

        MERGE (v)-[:HAS_GRAMMAR]->(g)

        MERGE (v)-[:HAS_VOCABULARY]->(vo)

        SET v.embedding = $embedding

        """,

        verse=item.get("verse", ""),

        meaning=item.get("meaning", ""),

        grammar=json.dumps(
            item.get("grammar", {}),
            ensure_ascii=False
        ),

        vocab=json.dumps(
            item.get("vocabulary", {}),
            ensure_ascii=False
        ),

        embedding=item.get("embedding", [])
        )

    # =====================================================
    # RUN PIPELINE
    # =====================================================

    def run(self):

        chunks = self.load_data()

        for chunk in chunks:

            items = self.parse_text(chunk)

            for item in items:

                if "verse" in item:

                    # =========================================
                    # EMBEDDING
                    # =========================================
                    embedding = self.generate_embedding(
                        item["verse"]
                    )

                    item["embedding"] = embedding

                    # =========================================
                    # INSERT GRAPH
                    # =========================================
                    self.insert_graph(item)

                    print(
                        f"✅ تم حفظ البيت: "
                        f"{item['verse'][:40]}..."
                    )

        self.driver.close()


# =====================================================
# FILES
# =====================================================

files = [

    "Dataset/معلقة الأعشى.txt",

    "Dataset/معلقة عنترة بن شداد.txt",

    "Dataset/معلقة لبيد بن ربيعة.txt"
]

# =====================================================
# RUN
# =====================================================

pipeline = PoetryPipeline(files)

pipeline.run()
