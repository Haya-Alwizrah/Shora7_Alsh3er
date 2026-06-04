import os
import re
from dotenv import load_dotenv
from openai import OpenAI
# pyrefly: ignore [missing-import]
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

load_dotenv(".env")


class GraphRag:

    def __init__(self):

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self.model_emb = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL")
        )

        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        try:
            self._create_indexes()
        except Exception as e:
            print(f"Warning during self-healing index verification: {e}")

    def _create_indexes(self):
        with self.driver.session() as session:
            try:
                # 1. Create Vector Index
                session.run("""
                CREATE VECTOR INDEX verse_vector IF NOT EXISTS
                FOR (v:Verse) ON (v.embedding)
                OPTIONS {indexConfig: {
                  `vector.dimensions`: 64,
                  `vector.similarity_function`: 'cosine'
                }}
                """)
            except Exception as e:
                print(f"Warning: Failed to create vector index: {e}")
                
            try:
                # 2. Create Fulltext Index
                session.run("""
                CREATE FULLTEXT INDEX verse_text IF NOT EXISTS
                FOR (v:Verse) ON EACH [v.text]
                """)
            except Exception as e:
                print(f"Warning: Failed to create fulltext index: {e}")

    def normalize_arabic(self, text):
        arabic_diacritics = re.compile(r"[\u064B-\u0652]")
        text = re.sub(arabic_diacritics, "", text)
        text = re.sub(r"[أإآ]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ة", "ه", text)
        return text.lower().strip()

    def extract_keywords(self, query):
        raw_stopwords = {
            "ما", "معنى", "كلمة", "شرح", "في", "من", "على", "عن", "البيت", "الذي", "التي", 
            "اعرب", "وضح", "فسر", "كيف", "من هو", "الشاعر", "قصيدة", "معلقة", "ماذا", "هل", 
            "أين", "يا", "لقد", "قد", "بين", "منها", "إليه", "عليه", "له", "لها", "بن", "ابن",
            "أعرب", "الموقع", "الإعرابي", "جملة", "إعراب", "الاعرابي", "لجملة",
            "الشعر", "تعني", "اذكر", "لماذا", "متى", "إبن", "هذا", "هذه", "ذلك", "تلك",
            "التي", "الذين", "هو", "هي", "أن", "إن", "كان", "يكون", "عند", "إلى",
            "اشرح", "فسّر", "وضّح", "بيّن", "اذكري"
        }
        stopwords = {self.normalize_arabic(w) for w in raw_stopwords}
        normalized_query = self.normalize_arabic(query)
        words = re.findall(r"\w+", normalized_query)
        keywords = [w for w in words if w not in stopwords and len(w) > 1]
        return keywords

    def _embed_query(self, query):
        return self.model_emb.encode(query).tolist()

    # 1. VECTOR SEARCH
    def vector_search(self, query, k=5):
        query_vector = self._embed_query(query)

        cypher = """
        CALL db.index.vector.queryNodes(
            'verse_vector',
            $k,
            $vector
        )
        YIELD node, score

        OPTIONAL MATCH (node)-[:HAS_MEANING]->(m)
        OPTIONAL MATCH (node)-[:HAS_VOCABULARY]->(vo)
        OPTIONAL MATCH (node)-[:HAS_GRAMMAR]->(g)
        OPTIONAL MATCH (p:Poet)-[:WROTE]->(node)

        RETURN
            node,
            score,
            p.name AS poet,
            m.text AS meaning,
            vo.text AS vocab,
            g.text AS grammar
        LIMIT $k
        """

        results = []
        try:
            with self.driver.session() as session:
                rows = session.run(cypher, vector=query_vector, k=k)
                for r in rows:
                    node = r["node"]
                    results.append({
                        "key": (r.get("poet", ""), node.get("verse_number", "")),
                        "poet": r.get("poet", ""),
                        "verse_number": node.get("verse_number", ""),
                        "verse": node.get("text", ""),
                        "meaning": r.get("meaning", ""),
                        "vocab": r.get("vocab", ""),
                        "grammar": r.get("grammar", ""),
                        "score": float(r.get("score", 0)),
                        "source": "vector"
                    })
        except Exception as e:
            print(f"Vector search failed: {e}")
        return results

    # 2. FULL TEXT SEARCH
    def keyword_search(self, query, k=5):
        keywords = self.extract_keywords(query)
        if keywords:
            lucene_query = " OR ".join([f'"{kw}"' for kw in keywords])
        else:
            lucene_query = query

        cypher = """
        CALL db.index.fulltext.queryNodes(
            'verse_text',
            $text_query  
        )
        YIELD node, score

        OPTIONAL MATCH (node)-[:HAS_MEANING]->(m)
        OPTIONAL MATCH (node)-[:HAS_VOCABULARY]->(vo)
        OPTIONAL MATCH (node)-[:HAS_GRAMMAR]->(g)
        OPTIONAL MATCH (p:Poet)-[:WROTE]->(node)

        RETURN
            node,
            score,
            p.name AS poet,
            m.text AS meaning,
            vo.text AS vocab,
            g.text AS grammar
        LIMIT $k
        """

        results = []
        try:
            with self.driver.session() as session:
                rows = session.run(cypher, text_query=lucene_query, k=k)
                for r in rows:
                    node = r["node"]
                    results.append({
                        "key": (r.get("poet", ""), node.get("verse_number", "")),
                        "poet": r.get("poet", ""),
                        "verse_number": node.get("verse_number", ""),
                        "verse": node.get("text", ""),
                        "meaning": r.get("meaning", ""),
                        "vocab": r.get("vocab", ""),
                        "grammar": r.get("grammar", ""),
                        "score": float(r.get("score", 0)),
                        "source": "keyword"
                    })
        except Exception as e:
            print(f"Keyword search failed: {e}")
        return results

    # 3. GRAPH SEARCH
    def graph_search(self, query, k=5):
        keywords = self.extract_keywords(query)
        if not keywords:
            return []

        cypher = """
        MATCH (v:Verse)
        OPTIONAL MATCH (p:Poet)-[:WROTE]->(v)
        OPTIONAL MATCH (v)-[:HAS_MEANING]->(m)
        OPTIONAL MATCH (v)-[:HAS_VOCABULARY]->(vo)
        OPTIONAL MATCH (v)-[:HAS_GRAMMAR]->(g)

        WITH v, p, m, vo, g,
             reduce(s = 0, word IN $keywords | 
               s + case when toLower(v.text) contains toLower(word) then 10 else 0 end
                 + case when toLower(coalesce(p.name, '')) contains toLower(word) then 5 else 0 end
                 + case when toLower(coalesce(m.text, '')) contains toLower(word) then 2 else 0 end
                 + case when toLower(coalesce(vo.text, '')) contains toLower(word) then 2 else 0 end
             ) AS score
        WHERE score > 0

        RETURN
            v,
            p.name AS poet,
            m.text AS meaning,
            vo.text AS vocab,
            g.text AS grammar,
            score
        ORDER BY score DESC
        LIMIT $k
        """

        results = []
        try:
            with self.driver.session() as session:
                rows = session.run(cypher, keywords=keywords, k=k)
                for r in rows:
                    node = r["v"]
                    results.append({
                        "key": (r.get("poet", ""), node.get("verse_number", "")),
                        "poet": r.get("poet", ""),
                        "verse_number": node.get("verse_number", ""),
                        "verse": node.get("text", ""),
                        "meaning": r.get("meaning", ""),
                        "vocab": r.get("vocab", ""),
                        "grammar": r.get("grammar", ""),
                        "score": float(r.get("score", 0)),
                        "source": "graph"
                    })
        except Exception as e:
            print(f"Graph search failed: {e}")
        return results

    # 4. HYBRID SEARCH
    def hybrid_search(self, query, k=3):
        pool_k = max(k * 4, 15)

        vector_results = self.vector_search(query, pool_k)
        keyword_results = self.keyword_search(query, pool_k)
        graph_results = self.graph_search(query, pool_k)

        all_items = {}
        for r in vector_results + keyword_results + graph_results:
            key = r["key"]
            if key not in all_items:
                all_items[key] = {
                    "poet": r["poet"],
                    "verse_number": r["verse_number"],
                    "verse": r["verse"],
                    "meaning": r["meaning"],
                    "vocab": r["vocab"],
                    "grammar": r["grammar"]
                }

        # RRF Rank Fusion
        rank_vector = [r["key"] for r in vector_results]
        rank_keyword = [r["key"] for r in keyword_results]
        rank_graph = [r["key"] for r in graph_results]

        rrf_scores = {}
        k_rrf = 60

        for key in all_items.keys():
            score = 0.0
            if key in rank_vector:
                score += 1.0 / (k_rrf + rank_vector.index(key) + 1)
            if key in rank_keyword:
                score += 1.2 / (k_rrf + rank_keyword.index(key) + 1)
            if key in rank_graph:
                score += 1.2 / (k_rrf + rank_graph.index(key) + 1)
            rrf_scores[key] = score

        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        ranked_results = []
        for key in sorted_keys[:k]:
            item = all_items[key]
            ranked_results.append({
                "text": f"""
البيت: {item['verse']}
الشاعر: {item['poet']}
المعنى: {item['meaning']}
المفردات: {item['vocab']}
الإعراب: {item['grammar']}
                """.strip(),
                "score": rrf_scores[key],
                "source": "hybrid"
            })

        return ranked_results

    def search_graph(self, query):

        results = self.hybrid_search(query, k=3)

        context = "\n\n-------\n\n".join(
            [r["text"] for r in results]
        )

        return context


    def generate_answer(self, query, context):

        system_prompt = """
        أنت خبير في الشعر العربي (المعلقات).
        أجب فقط من السياق المقدم.
        إذا لا يوجد جواب قل: لا أعلم.
        """

        user_prompt = f"""
        Context:
        {context}

        Question:
        {query}

        Answer in clear Arabic:
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

    def ask(self, query):

        context = self.search_graph(query)

        if not context:
            return "لا توجد بيانات", ""

        answer = self.generate_answer(query, context)

        return answer, context


    def close(self):
        self.driver.close()


if __name__ == "__main__":

    rag = GraphRag()

    # q = "اشرح بيت (يا دار عبلة بالجواء تكلمي) شرحاً وافياً."

    # answer, context = rag.ask(q)

    # print(answer)

    rag.close()

