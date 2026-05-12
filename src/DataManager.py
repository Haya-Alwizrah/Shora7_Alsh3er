import os
import re
import chromadb
from chromadb.utils import embedding_functions

class DataManager:

    def __init__(self, files, db_path, collection_name, embed_model):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name

        self.files = files
        self.chunks = []
        self.metadatas = []

        self.embed_model=embed_model
        self.embedding_func = None
        self.collection = None        

    def get_embedding_func(self):
        if self.embedding_func is None:
            print("Loading embedding model...")
            self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embed_model)
        
        return self.embedding_func
    
    def split_poem_blocks(self, text):
        chunks = re.split(r"\n[-]{10,}\n",text)

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def load_files(self):
        for file_path in self.files:
            with open(file_path , "r", encoding="utf-8") as f:
                text = f.read()

            chunks = self.split_poem_blocks(text)

            for i, chunk in enumerate(chunks):
                self.chunks.append(chunk)
                self.metadatas.append({
                    "source": file_path ,
                    "chunk_id": i
                })

        print(len(self.chunks))
    
    def create_collection(self):
        self.collection = self.client.get_or_create_collection(
            name = self.collection_name,
            embedding_function = self.get_embedding_func()
        )

        print("collection created")

    def db_exists(self):
        try:
            collections = self.client.list_collections()
            return any(c.name == self.collection_name for c in collections)
        except:
            return False
        
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

        print("Documents added to vector DB")

    def prepare(self):
        if self.db_exists():
            print("DB already exists → skipping embedding step")
            self.create_collection()
            return
        
        print("Building DB from scratch...")

        self.load_files()
        self.create_collection()
        self.add_documents()

    def search(self, query, n_results=1):

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        return results["documents"][0]