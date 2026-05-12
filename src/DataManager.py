import os
import re
import chromadb
from chromadb.utils import embedding_functions

class DataManager:

    def __init__(self, folder_path, db_path, collection_name, embed_model):
        self.folder_path = folder_path
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.collection = None
        self.chunks = []
        self.metadatas = []

    def split_poem_blocks(self, text):
        chunks = re.split(r"\n[-]{10,}\n",text)

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def load_files(self):
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.folder_path, filename)

                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = self.split_poem_blocks(text)

                for i, chunk in enumerate(chunks):
                    self.chunks.append(chunk)
                    self.metadatas.append({
                        "source": filename,
                        "chunk_id": i
                    })
    
    def create_collection(self):

        if self.collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(self.collection_name)

        self.collection = self.client.create_collection(
            name = self.collection_name,
            embedding_function = self.embedding_func
        )

    def add_documents(self, batch_size=64):

        for i in range(0, len(self.chunks), batch_size):

            batch_chunks = self.chunks[i:i+batch_size]
            batch_metadatas = self.metadatas[i:i+batch_size]
            batch_ids = [str(x) for x in range(i, i + len(batch_chunks))]

            self.collection.add(
                documents = batch_chunks,
                metadatas = batch_metadatas,
                ids = batch_ids
            )

    def prepare(self):
        self.load_files()
        self.create_collection()
        self.add_documents()

    def search(self, query, n_results=1):

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        return results["documents"][0]