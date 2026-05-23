import os
from dotenv import load_dotenv
from openai import OpenAI
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


    def _embed_query(self, query):
        return self.model_emb.encode(query).tolist()

    # 1. VECTOR SEARCH
    def vector_search(self, query, k=2):

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

        RETURN
            node,
            score,
            m.text AS meaning,
            vo.text AS vocab,
            g.text AS grammar
        """

        results = []

        with self.driver.session() as session:

            rows = session.run(cypher, vector=query_vector, k=k)

            for r in rows:

                node = r["node"]

                results.append({
                    "text": f"""
                    البيت:
                    {node.get('text', '')}

                    المعنى:
                    {r.get('meaning', '')}

                    المفردات:
                    {r.get('vocab', '')}

                    الإعراب:
                    {r.get('grammar', '')}
                    """.strip(),

                    "score": float(r.get("score", 0)),  

                    "source": "vector"
                })

        return results

    # 2. FULL TEXT SEARCH
    def keyword_search(self, query, k=2):

        cypher = """
        CALL db.index.fulltext.queryNodes(
            'verse_text',
            $text_query  
        )
        YIELD node, score

        OPTIONAL MATCH (node)-[:HAS_MEANING]->(m)
        OPTIONAL MATCH (node)-[:HAS_VOCABULARY]->(vo)
        OPTIONAL MATCH (node)-[:HAS_GRAMMAR]->(g)

        RETURN
            node,
            score,
            m.text AS meaning,
            vo.text AS vocab,
            g.text AS grammar

        LIMIT $k
        """

        results = []

        with self.driver.session() as session:

            rows = session.run(
                cypher,
                text_query=query,  
                k=k
            )

            for r in rows:

                node = r["node"]

                results.append({
                    "text": f"""
                    البيت:
                    {node.get('text', '')}

                    المعنى:
                    {r.get('meaning', '')}

                    المفردات:
                    {r.get('vocab', '')}

                    الإعراب:
                    {r.get('grammar', '')}
                    """.strip(),

                    "score": float(r.get("score", 0)),  

                    "source": "keyword"
                })

        return results

    # 3. GRAPH SEARCH
    def graph_search(self, query, k=2):

        cypher = """
        MATCH (v:Verse)

        WHERE toLower(v.text) CONTAINS toLower($text_query)  
        // ✔️ FIX: improved matching (case insensitive)

        OPTIONAL MATCH (v)-[:HAS_MEANING]->(m)
        OPTIONAL MATCH (v)-[:HAS_VOCABULARY]->(vo)
        OPTIONAL MATCH (v)-[:HAS_GRAMMAR]->(g)

        RETURN
            v.text AS verse,
            m.text AS meaning,
            vo.text AS vocab,
            g.text AS grammar

        LIMIT $k
        """

        results = []

        with self.driver.session() as session:

            rows = session.run(
                cypher,
                text_query=query,  
                k=k
            )

            for r in rows:

                results.append({
                    "text": f"""
                    البيت:
                    {r.get('verse', '')}

                    المعنى:
                    {r.get('meaning', '')}

                    المفردات:
                    {r.get('vocab', '')}

                    الإعراب:
                    {r.get('grammar', '')}
                    """.strip(),

                    "score": 0.9,

                    "source": "graph"
                })

        return results

    # 4. HYBRID SEARCH
    def hybrid_search(self, query, k=2):

        vector_results = self.vector_search(query, k)
        keyword_results = self.keyword_search(query, k)
        graph_results = self.graph_search(query, k)

        all_results = vector_results + keyword_results + graph_results

        # RERANKING
        def weight(r):
            base = float(r.get("score", 0))  

            if r["source"] == "vector":
                return base * 1.0
            elif r["source"] == "keyword":
                return base * 1.2
            else:
                return base * 1.1

        ranked = sorted(all_results, key=weight, reverse=True)

        return ranked[:k]

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

