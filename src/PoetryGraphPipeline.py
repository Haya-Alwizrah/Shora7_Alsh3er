from datasets import load_dataset
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import os
import json


class PoetryPipeline:

    def __init__(self):

        load_dotenv()

        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(
                os.getenv("NEO4J_USERNAME"),
                os.getenv("NEO4J_PASSWORD")
            )
        )

        # Arabic embedding model
        self.embedding_model = SentenceTransformer(SentenceTransformer(os.getenv("EMBEDDING_MODEL")))

        # Load dataset from HuggingFace
        self.dataset = load_dataset(
            "SarahALo/The-Ten-Muallaqat-Dataset"
        )


    def generate_embedding(self, text):

        return self.embedding_model.encode(text).tolist()


    def insert_graph(self, item):

        with self.driver.session() as session:

            session.execute_write(
                self._create_nodes,
                item
            )

    @staticmethod
    def _create_nodes(tx, item):

        tx.run("""

        MERGE (p:Poet {
            name: $poet
        })

        MERGE (v:Verse {
            verse_number: $verse_number,
            text: $verse
        })

        MERGE (m:Meaning {
            text: $meaning
        })

        MERGE (g:Grammar {
            text: $grammar
        })

        MERGE (vo:Vocabulary {
            text: $vocabulary
        })

        MERGE (p)-[:WROTE]->(v)

        MERGE (v)-[:HAS_MEANING]->(m)

        MERGE (v)-[:HAS_GRAMMAR]->(g)

        MERGE (v)-[:HAS_VOCABULARY]->(vo)

        SET v.embedding = $embedding

        """,

        poet=item.get("poet", ""),

        verse_number=item.get("verse_number", ""),

        verse=item.get("verse", ""),

        meaning=item.get("meaning", ""),

        grammar=item.get("grammar", ""),

        vocabulary=item.get("vocabulary", ""),

        embedding=item.get("embedding", [])
        )

    def process_split(self, split_name):

        print(f"\nProcessing {split_name} split...\n")

        split_data = self.dataset[split_name]

        for row in split_data:

            try:

                item = {

                    "poet": row.get("الشاعر", ""),

                    "verse_number": row.get("رقم البيت", ""),

                    "verse": row.get("البيت", ""),

                    "vocabulary": row.get("المفردات", ""),

                    "meaning": row.get("المعنى", ""),

                    "grammar": row.get("الإعراب", "")
                }

                # embedding from verse only
                embedding = self.generate_embedding(
                    item["verse"]
                )

                item["embedding"] = embedding

                # save to neo4j
                self.insert_graph(item)

                print(
                    f" Saved verse {item['verse_number']} "
                    f"for {item['poet']}"
                )

            except Exception as e:

                print(f" Error: {e}")


    def run(self):

        # Process train
        self.process_split("train")

        # Process test
        self.process_split("test")

        self.driver.close()

        print("\n Pipeline completed successfully")


pipeline = PoetryPipeline()

pipeline.run()