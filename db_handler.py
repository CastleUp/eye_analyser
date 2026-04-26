import chromadb
from chromadb.config import Settings
import uuid

class DBHandler:
    def __init__(self, db_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        # Create or get collection
        # We use cosine similarity as requested in TZ
        self.collection = self.client.get_or_create_collection(
            name="eye_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

    def add_user(self, name, embedding):
        # embedding should be a 1D numpy array or list
        if isinstance(embedding, (list, tuple)):
            embedding_list = list(embedding)
        else:
            embedding_list = embedding.tolist()
            
        self.collection.add(
            embeddings=[embedding_list],
            metadatas=[{"name": name}],
            ids=[str(uuid.uuid4())]
        )

    def query_user(self, embedding, threshold=0.4):
        if embedding is None:
            return None, 1.0
            
        if not isinstance(embedding, list):
            embedding_list = embedding.tolist()
        else:
            embedding_list = embedding
            
        results = self.collection.query(
            query_embeddings=[embedding_list],
            n_results=1
        )
        
        if not results['ids'][0]:
            return None, 1.0
            
        distance = results['distances'][0][0]
        name = results['metadatas'][0][0]['name']
        
        if distance < threshold:
            return name, distance
        else:
            return None, distance

    def clear_database(self):
        try:
            self.client.delete_collection("eye_embeddings")
        except:
            pass
        self.collection = self.client.get_or_create_collection(
            name="eye_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

    def delete_all(self):
        # Utility for testing
        results = self.collection.get()
        ids = results['ids']
        if ids:
            self.collection.delete(ids=ids)
