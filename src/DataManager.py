import os
import re
import chromadb
from chromadb.utils import embedding_functions


class DataManager:
    def __init__(self, data_source, db_path, collection_name, EMBEDDING_MODEL):

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.embed_model = EMBEDDING_MODEL
        
        if isinstance(data_source, str) and os.path.isdir(data_source):
            self.files = [os.path.join(data_source, f) for f in os.listdir(data_source) if f.endswith('.txt')]
        else:
            self.files = data_source

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
    
    def split_poem_blocks(self, text):
        chunks = re.split(r"\n[-]{10,}\n", text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def load_files(self):
        if not self.files:
            print("Warning: No files found to load.")
            return

        for file_path in self.files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = self.split_poem_blocks(text)

                for i, chunk in enumerate(chunks):
                    self.chunks.append(chunk)
                    self.metadatas.append({
                        "source": os.path.basename(file_path),
                        "chunk_id": i
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"Total chunks loaded: {len(self.chunks)}")
    
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
            batch_ids = [f"{m['source']}_{m['chunk_id']}" for m in batch_metadatas]

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
