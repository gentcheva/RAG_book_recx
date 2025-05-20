import faiss
from data_embedding import *

# Get the dimensionality of your embeddings (e.g., 768 for all-mpnet-base-v2)
embedding_dimension = book_embeddings_np.shape[1]

# Create an IndexFlatL2 index
# This index simply stores the vectors and uses brute-force L2 distance search.
index = faiss.IndexFlatL2(embedding_dimension)

print(f"FAISS index created with dimension: {index.d}")
print(f"Is the index trained? {index.is_trained}")

index.add(book_embeddings_np)

print(f"Number of vectors in the index: {index.ntotal}")