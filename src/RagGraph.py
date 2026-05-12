import os
import torch
import re
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

load_dotenv(".env")

class GraphRag:
    def __init__(self):
        self.client = OpenAI(
        #     base_url="https://openrouter.ai/api/v1",
        #     api_key=os.getenv("OPENROUTER_API_KEY")
        # )
        api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model_name ="gpt-4o-mini"

        # self.model_name = "qwen/qwen3-next-80b-a3b-instruct:free"
        self.model_emb = SentenceTransformer("Sarah0001/Arabic_embed_model")
        
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def _embed_query(self, query):
        return self.model_emb.encode(query).tolist()

    def search_graph(self, query_text, k=2):
        query_vector = self._embed_query(query_text)
        
        cypher_query = """
        CALL db.index.vector.queryNodes('verse_vector', $k, $queryVector)
        YIELD node, score
        RETURN node.text AS verse, node.meaning AS meaning, 
               node.grammar AS grammar, node.vocabulary AS vocab
        """
        
        context_parts = []
        with self.driver.session() as session:
            result = session.run(cypher_query, queryVector=query_vector, k=k)
            for record in result:
                context_parts.append(
                    f"Verse: {record['verse']}\n"
                    f"Meaning: {record['meaning']}\n"
                    f"Grammar: {record['grammar']}\n"
                    f"Vocabulary: {record['vocab']}\n"
                    f"---"
                )
        return "\n".join(context_parts)

    def generate_answer(self, query):
        context = self.search_graph(query)

        if not context:
            return "No relevant information found in the database."

        system_prompt = "You are an expert in Arabic poetry and Mu'allaqat. Answer based only on the provided context."
        user_prompt = f"""Context:
        {context}

        Question: {query}
        Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    rag = GraphRag()
    rag.close()