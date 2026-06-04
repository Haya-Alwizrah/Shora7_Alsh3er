import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
import re


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

    def normalize_arabic(self, text):
        # Remove diacritics (harakat)
        arabic_diacritics = re.compile(r"[\u064B-\u0652]")
        text = re.sub(arabic_diacritics, "", text)
        
        # Normalize letters
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

    def _get_verses_cache(self):
        if not hasattr(self, "_verses_cache") or not self._verses_cache:
            self._verses_cache = {}
            if self.collection is None:
                self._create_collection()
            
            # Fetch all documents in local Chroma collection
            all_docs = self.collection.get()
            
            if all_docs and "documents" in all_docs and all_docs["documents"]:
                for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
                    poet = meta.get("poet", "")
                    v_num = str(meta.get("verse_number", ""))
                    v_type = meta.get("type", "")
                    
                    key = (poet, v_num)
                    if key not in self._verses_cache:
                        self._verses_cache[key] = {
                            "poet": poet,
                            "verse_number": v_num,
                            "verse": "",
                            "vocabulary": "",
                            "meaning": "",
                            "grammar": ""
                        }
                    
                    if v_type == "vocabulary":
                        parts = doc.split(", المفردات: ")
                        if len(parts) >= 2:
                            self._verses_cache[key]["verse"] = parts[0].replace("البيت: ", "").strip()
                            self._verses_cache[key]["vocabulary"] = parts[1].strip()
                    elif v_type == "meaning":
                        parts = doc.split(", المعنى: ")
                        if len(parts) >= 2:
                            self._verses_cache[key]["verse"] = parts[0].replace("البيت: ", "").strip()
                            self._verses_cache[key]["meaning"] = parts[1].strip()
                    elif v_type == "grammar":
                        parts = doc.split(", الاعراب: ")
                        if len(parts) >= 2:
                            self._verses_cache[key]["verse"] = parts[0].replace("البيت: ", "").strip()
                            self._verses_cache[key]["grammar"] = parts[1].strip()
                            
        return self._verses_cache

    def _lexical_search(self, keywords):
        verses_cache = self._get_verses_cache()
        lexical_results = []
        
        for key, item in verses_cache.items():
            searchable_text = f"{item['poet']} {item['verse']} {item['meaning']} {item['vocabulary']} {item['grammar']}"
            normalized_text = self.normalize_arabic(searchable_text)
            
            match_count = 0
            for kw in keywords:
                if kw in normalized_text:
                    norm_verse = self.normalize_arabic(item['verse'])
                    if kw in norm_verse:
                        match_count += 3
                    else:
                        match_count += 1
                        
            if match_count > 0:
                lexical_results.append({
                    "key": key,
                    "score": match_count,
                    "item": item
                })
                
        return sorted(lexical_results, key=lambda x: x["score"], reverse=True)

    def search(self, query, n_results=1):
        if self.collection is None:
            self._create_collection()
            
        verses_cache = self._get_verses_cache()
        keywords = self.extract_keywords(query)
        
        # 1. Vector Search Candidates — large pool to avoid missing relevant verses
        total_count = self.collection.count() or 10
        chroma_n = min(80, total_count)
        
        results = self.collection.query(
            query_texts=[query],
            n_results=chroma_n
        )
        
        semantic_scores = {}
        if results and "documents" in results and results["documents"] and len(results["documents"][0]) > 0:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.5] * len(documents)
            
            for doc, meta, dist in zip(documents, metadatas, distances):
                poet = meta.get("poet", "")
                v_num = str(meta.get("verse_number", ""))
                key = (poet, v_num)
                similarity = 1.0 / (1.0 + dist)
                
                if key not in semantic_scores or similarity > semantic_scores[key]:
                    semantic_scores[key] = similarity
                    
        # 2. Lexical Search
        lexical_results = self._lexical_search(keywords)
        lexical_scores = {res["key"]: res["score"] for res in lexical_results}
        
        # 3. Fusion and Reranking using RRF
        all_keys = set(semantic_scores.keys()).union(set(lexical_scores.keys()))
        
        if not all_keys:
            return "عذرًا، لم يتم العثور على أي نصوص مطابقة لسؤالك."
        
        # Build rank dicts for O(1) lookup instead of O(n) .index()
        ranked_semantic = sorted(semantic_scores.keys(), key=lambda k: semantic_scores[k], reverse=True)
        ranked_lexical = sorted(lexical_scores.keys(), key=lambda k: lexical_scores[k], reverse=True)
        
        sem_rank_map = {k: i + 1 for i, k in enumerate(ranked_semantic)}
        lex_rank_map = {k: i + 1 for i, k in enumerate(ranked_lexical)}
        
        rrf_scores = {}
        k_rrf = 60
        
        for key in all_keys:
            rrf_score = 0.0
            if key in sem_rank_map:
                rrf_score += 1.0 / (k_rrf + sem_rank_map[key])
            if key in lex_rank_map:
                rrf_score += 1.2 / (k_rrf + lex_rank_map[key])
            
            # Direct Match Bonus: reward verses where query keywords
            # appear literally in the verse text (strong relevance signal)
            if keywords:
                item = verses_cache.get(key)
                if item and item.get("verse"):
                    norm_verse = self.normalize_arabic(item["verse"])
                    direct_hits = sum(1 for kw in keywords if kw in norm_verse)
                    if direct_hits > 0:
                        rrf_score += 0.015 * direct_hits
            
            rrf_scores[key] = rrf_score
            
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        
        # We enrich contexts: return the top match details
        # By default we retrieve top 2 verses to give context for standard RAG
        limit = max(n_results, 2)
        top_keys = sorted_keys[:limit]
        
        formatted_contexts = []
        for key in top_keys:
            item = verses_cache.get(key)
            if item:
                v_text = item["verse"] or "غير متوفر"
                v_poet = item["poet"] or "غير متوفر"
                v_meaning = item["meaning"] or "غير متوفر"
                v_vocab = item["vocabulary"] or "غير متوفر"
                v_grammar = item["grammar"] or "غير متوفر"
                
                formatted_contexts.append(f"""
البيت: {v_text}
الشاعر: {v_poet}
المعنى: {v_meaning}
المفردات: {v_vocab}
الإعراب: {v_grammar}
                """.strip())
                
        return "\n\n====================\n\n".join(formatted_contexts)
