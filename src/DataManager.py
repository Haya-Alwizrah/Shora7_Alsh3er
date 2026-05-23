import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset


class DataManager:
    def __init__(self, dataset_name, db_path, collection_name, EMBEDDING_MODEL):

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.embed_model = EMBEDDING_MODEL
        
        self.dataset_name = dataset_name

        self.chunks = []
        self.metadatas = []
        self.embedding_func = None
        self.collection = None        

    def get_embedding_func(self):
        if self.embedding_func is None:
            self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embed_model
            )
        return self.embedding_func
    
    def load_files(self):
        print(f"Loading dataset from HuggingFace: {self.dataset_name}...")
        dataset = load_dataset(self.dataset_name)
        
        all_splits = ['train', 'test']
        
        for split in all_splits:
            if split in dataset:
                for row in dataset[split]:
                    
                    poet = row.get("الشاعر", "") or ""
                    verse_number = row.get("رقم البيت", "") or ""
                    verse = row.get("البيت", "") or ""
                    vocabulary = row.get("المفردات", "") or "غير متوفر"
                    meaning = row.get("المعنى", "") or "غير متوفر"
                    grammar = row.get("الاعراب", "") or "غير متوفر"

                    chunk = f"""
                    الشاعر: {poet}
                    البيت: {verse}
                    المفردات: {vocabulary}
                    المعنى: {meaning}
                    الاعراب: {grammar}
                    """.strip()

                    self.chunks.append(chunk)
                    self.metadatas.append({
                        "الشاعر": poet,
                        "رقم البيت": str(verse_number)  
                    })
    def create_collection(self):
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.get_embedding_func()
        )
        print(f"Collection '{self.collection_name}' is ready.")

    def db_exists(self):
        try:
            collections = self.client.list_collections()
            return any(c.name == self.collection_name for c in collections)
        except:
            return False
        
    def add_documents(self, batch_size=64):
        if not self.chunks:
            return

        for i in range(0, len(self.chunks), batch_size):
            batch_chunks = self.chunks[i:i+batch_size]
            batch_metadatas = self.metadatas[i:i+batch_size]
            batch_ids = [f"{m['الشاعر']}_{m['رقم البيت']}" for m in batch_metadatas]

            self.collection.add(
                documents=batch_chunks,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        print("Documents added to vector DB successfully.")

    def prepare(self):
        if self.db_exists():
            print("DB already exists → Loading existing collection.")
            self.create_collection()
            if self.collection.count() == 0:
                self._build_from_scratch()
        else:
            self._build_from_scratch()

    def _build_from_scratch(self):
        print("Building DB from scratch")
        self.load_files()
        self.create_collection()
        self.add_documents()

    def search(self, query, n_results=2):
        if not self.collection:
            self.create_collection()
            
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results["documents"][0]
