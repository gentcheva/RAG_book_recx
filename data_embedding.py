import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from book_data import book_data

model = SentenceTransformer('all-mpnet-base-v2')
book_embeddings = []
for book in book_data:
    summary = book.get("summary")
    if summary:
        embedding = model.encode(summary)
        book_embeddings.append(embedding)
    else:
        book_embeddings.append(None)

book_embeddings_np = np.array(book_embeddings).astype(np.float32)

print(f"Shape of the combined embeddings array: {book_embeddings_np.shape}")
print(f"Number of embeddings generated: {len(book_embeddings)}")
print(f"Shape of the first embedding: {book_embeddings[0].shape if book_embeddings[0] is not None else None}")
print(f"Shape of the combined embeddings array for FAISS: {book_embeddings_np.shape}")

embedding_dimension = book_embeddings_np.shape[1]
index = faiss.IndexFlatL2(embedding_dimension) # Create the index
index.add(book_embeddings_np) # Add the embeddings

print(f"FAISS index created and populated with {index.ntotal} vectors.")
