import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset


class DataManager:
    def __init__(self, dataset_name, db_path, collection_name, EMBEDDING_MODEL):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.dataset_name = dataset_name
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name= EMBEDDING_MODEL
        )

        self.chunks = []
        self.metadatas = []
        self.collection = None

    def _load_files(self):
        print(f"Loading dataset from HuggingFace: {self.dataset_name}...")
        dataset = load_dataset(self.dataset_name)
        
        print("start chunking...")
        for split in dataset:
            for row in dataset[split]:
                    
                poet = row.get("الشاعر", "")
                verse_number = row.get("رقم البيت", "") or ""

                verse = row.get("البيت", "") or ""
                vocabulary = row.get("المفردات", "") or ""
                meaning = row.get("المعنى", "") or ""
                grammar = row.get("الاعراب", "") or ""

                self.chunks.append(f"البيت: {verse}, المفردات: {vocabulary}")
                self.metadatas.append({
                    "type": "vocabulary",
                    "poet": poet,
                    "verse_number": str(verse_number),
                })

                self.chunks.append(f"البيت: {verse}, المعنى: {meaning}")
                self.metadatas.append({
                    "type": "meaning",
                    "poet": poet,
                    "verse_number": str(verse_number),
                })

                self.chunks.append(f"البيت: {verse}, الاعراب: {grammar}")
                self.metadatas.append({
                    "type": "grammar",
                    "poet": poet,
                    "verse_number": str(verse_number),
                })
                    
    def _create_collection(self):
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_func
        )
        print(f"Collection '{self.collection_name}' is ready.")
        
    def _add_documents(self, batch_size=64):
        if not self.chunks:
            return

        for i in range(0, len(self.chunks), batch_size):
            batch_chunks = self.chunks[i:i+batch_size]
            batch_metadatas = self.metadatas[i:i+batch_size]
            batch_ids = [f"{m['poet']}_{m['verse_number']}_{m['type']}" for m in batch_metadatas]

            self.collection.add(
                documents=batch_chunks,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        print("Documents added to vector DB successfully.")

#---------------------------------------------------------------------------------------------------------------------------------------------------------
    def prepare(self):
        print("creat or get collection")
        self._create_collection()

        if self.collection.count() == 0:
            print("Empty collection → Building DB")
            self._load_files()
            self._add_documents()
        else:
            print("Existing collection loaded.")

    def search(self, query, n_results=1):
        if self.collection is None:
            self._create_collection()
            
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        return "\n\n".join(results["documents"][0])
