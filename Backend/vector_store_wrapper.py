import os
import uuid
from typing import List, Optional, Dict, Any

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone

class VectorStoreWrapper:
    def __init__(self, index_name: str = "legaladviser", api_key: str | None = None):
        self.pc = Pinecone(api_key=api_key or os.getenv("PINECONE_API_KEY"))
        index_names =  [i.get("name") for i in self.pc.list_indexes().indexes]
        if index_name not in index_names:
            raise f"Cannot find pinecone index {index_name}"
        self.index = self.pc.Index(index_name)

    def upsert(
            self,
            embeddings: List[List[float]],
            metadatas: Optional[List[Dict[str, Any]]] = None,
            ids: Optional[List[str]] = None,
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]
        if metadatas is None:
            metadatas = [{} for _ in embeddings]

        vectors = [
            {"id": id_, "values": vector, "metadata": meta}
            for id_, vector, meta in zip(ids, embeddings, metadatas)
        ]

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

        return ids

    def query(self, vector: list[float] | None = None, id: str | None = None, top_k: int = 5, namespace: str | None = None, filter: dict | None = None, include_metadata: bool = True):
        return self.index.query(vector=vector, id=id, top_k=top_k, namespace=namespace or "", filter=filter, include_metadata=include_metadata)

    def delete(self, ids: list[str] | None = None, namespace: str | None = None, delete_all: bool = False,
               filter: dict | None = None):
        return self.index.delete(ids=ids, namespace=namespace or "", delete_all=delete_all, filter=filter)

def test():
    dim = 768
    wrapper = VectorStoreWrapper()
    vector = np.random.rand(dim).tolist()
    metadata = {"genre": "demo"}
    vec_id = "test-id"

    # Upsert using the new method
    returned_ids = wrapper.upsert([vector], metadatas=[metadata], ids=[vec_id])
    assert returned_ids == [vec_id]

    # Query for nearest neighbour
    query_res = wrapper.query(vector=vector, top_k=1)
    top_match = query_res["matches"][0]
    assert top_match["id"] == vec_id
    assert np.isclose(top_match["score"], 1.0, atol=1e-3)
    assert top_match.get("metadata", {}).get("genre") == "demo"

    print("upsert_vectors and query test passed for 768‑dimensional vector with metadata.")

if __name__ == "__main__":
    load_dotenv()
    test()
