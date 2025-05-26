import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from book_data import book_data

def prepare_vector_database(books_data, model_name='all-mpnet-base-v2'):
    model = SentenceTransformer(model_name)
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

    return index, book_embeddings_np, model


# --- 2. Retrieval Function ---

def retrieve_books(user_query, faiss_index, original_book_data, embedding_model, num_results=3):
    """
    Retrieves the most semantically similar books based on a user query.

    Args:
        user_query (str): The query provided by the user.
        faiss_index (faiss.Index): The pre-built FAISS index containing book embeddings.
        original_book_data (list of dict): The original list of book data,
                                            used to map FAISS indices back to book details.
        embedding_model (SentenceTransformer): The same model used to embed the book summaries.
        num_results (int): The number of top similar books to retrieve.

    Returns:
        list of dict: A list of dictionaries, each representing a retrieved book
                      with its full details (title, author, summary, etc.).
    """
    print(f"\nEmbedding user query: '{user_query}'...")
    # 1. Embed the user's query using the same model
    query_embedding = embedding_model.encode(user_query, convert_to_numpy=True)

    # 2. Reshape and convert to float32 for FAISS
    # FAISS expects a 2D array, even for a single query (batch_size, embedding_dim)
    query_embedding_2d = np.array([query_embedding]).astype('float32')
    print(f"Query embedding shape for FAISS search: {query_embedding_2d.shape}")

    # 3. Perform a similarity search in the FAISS index
    print(f"Searching FAISS index for top {num_results} results...")
    distances, indices = faiss_index.search(query_embedding_2d, num_results)

    retrieved_books = []
    # 4. Extract and return the actual book information
    # indices is a 2D array, e.g., [[idx1, idx2, idx3]]
    # distances is also a 2D array, e.g., [[dist1, dist2, dist3]]

    # We loop through the first (and only) row of the indices array
    for i, idx in enumerate(indices[0]):  # indices[0] gives the 1D array of actual indices - idx is the original index of that book
        # Ensure the index is valid and corresponds to an actual book
        if 0 <= idx < len(original_book_data):
            book_details = original_book_data[idx]
            retrieved_books.append(book_details)
            print(f"  Retrieved book: '{book_details.get('title', 'Unknown')}' (Distance: {distances[0][i]:.4f})")
        else:
            print(f"  Warning: Invalid index {idx} returned by FAISS. Skipping.")

    print("Retrieval complete.")
    return retrieved_books


# --- 3. Main Execution (Demonstration of the full flow) ---

if __name__ == "__main__":
    print("--- Starting RAG System Setup ---")

    # Prepare the vector database (embedding and FAISS indexing)
    faiss_index, _, embedding_model = prepare_vector_database(book_data)

    print("\n--- FAISS Index Ready ---")

    # Define a dummy user query
    user_query = "I'm looking for a high fantasy adventure with elves and dragons."
    # user_query = "Suggest a dark fantasy book with political intrigue." # Uncomment to test another query

    # Retrieve relevant books
    num_recommendations = 3
    retrieved_books = retrieve_books(user_query, faiss_index, book_data, embedding_model,
                                     num_results=num_recommendations)

    print(f"\n--- Top {len(retrieved_books)} Books Retrieved for Query: '{user_query}' ---")
    if retrieved_books:
        for i, book in enumerate(retrieved_books):
            print(f"{i + 1}. Title: {book.get('title', 'N/A')}")
            print(f"   Author: {book.get('author', 'N/A')}")
            print(f"   Genre: {book.get('genre', 'N/A')}")
            print(f"   Summary: {book.get('summary', 'N/A')[:150]}...")  # Print first 150 chars of summary
            print("-" * 20)
    else:
        print("No books retrieved. Check your data or query.")

    print("\n--- Retrieval Process Complete. Next: Generation with LLM ---")

