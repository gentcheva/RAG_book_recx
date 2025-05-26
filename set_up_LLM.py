import openai
import os # To access environment variables
from dotenv import load_dotenv
load_dotenv()

# Load environment variables from .env file at the very beginning


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
    # user_query = "I'm looking for a high fantasy adventure with elves and dragons."
    # # user_query = "Suggest a dark fantasy book with political intrigue." # Uncomment to test another query
    #
    # # Retrieve relevant books
    # num_recommendations = 3
    # retrieved_books = retrieve_books(user_query, faiss_index, book_data, embedding_model,
    #                                  num_results=num_recommendations)
    #
    # print(f"\n--- Top {len(retrieved_books)} Books Retrieved for Query: '{user_query}' ---")
    # if retrieved_books:
    #     for i, book in enumerate(retrieved_books):
    #         print(f"{i + 1}. Title: {book.get('title', 'N/A')}")
    #         print(f"   Author: {book.get('author', 'N/A')}")
    #         print(f"   Genre: {book.get('genre', 'N/A')}")
    #         print(f"   Summary: {book.get('summary', 'N/A')[:150]}...")  # Print first 150 chars of summary
    #         print("-" * 20)
    # else:
    #     print("No books retrieved. Check your data or query.")
    #
    # print("\n--- Retrieval Process Complete. Next: Generation with LLM ---")



# --- 3. Function to Generate Recommendations using LLM (OpenAI Version) ---

def generate_recommendations_openai(user_query, retrieved_books, llm_model_name="gpt-3.5-turbo"):
    """
    Generates book recommendations and explanations using an OpenAI LLM,
    augmented by retrieved book information.

    Args:
        user_query (str): The original query from the user.
        retrieved_books (list of dict): A list of dictionaries, each containing
                                        details of a relevant book.
        llm_model_name (str): The name of the OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4").

    Returns:
        str: A formatted string containing the generated book recommendations and explanations.
    """
    # Configure the OpenAI API with your API key
    # It's crucial to set OPENAI_API_KEY as an environment variable
    # DO NOT hardcode your API key here!
    try:
        # Use os.getenv() which is slightly safer than os.environ[] as it returns None if not found
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY not found in environment variables or .env file. Please set it."
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        return f"Error initializing OpenAI client: {e}"

    # try:
    #     client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # except KeyError:
    #     return "Error: OPENAI_API_KEY environment variable not set. Please set it to your OpenAI API key."
    # except Exception as e:
    #     return f"Error initializing OpenAI client: {e}"

    # --- Prompt Engineering using Chat Message Format ---
    # OpenAI Chat Completions API uses a list of messages with roles:
    # 'system': Sets the behavior/persona of the assistant.
    # 'user': The user's input.
    # 'assistant': Previous responses from the assistant (for multi-turn conversations).

    # System message: Sets the context and persona for the LLM
    messages = [
        {"role": "system", "content": (
            "You are an expert book recommender. Your goal is to provide helpful and creative book suggestions "
            "based on user queries and relevant books found in a database. "
            "You should explain why each recommendation is suitable based on the user's request and the provided context. "
            "Always present recommendations in a clear, easy-to-read list format. "
            "Ensure you do NOT recommend the exact books that were provided as 'retrieved context'; instead, use them as inspiration for new, similar suggestions."
        )}
    ]

    # User message: Combine the user's query and the retrieved context
    user_content = f"The user is looking for book recommendations based on this request: '{user_query}'\n\n"

    if retrieved_books:
        user_content += "I have found the following relevant books based on the user's query. Please use these as context to inform your recommendations:\n\n"
        for i, book in enumerate(retrieved_books):
            title = book.get('title', 'Unknown Title')
            author = book.get('author', 'Unknown Author')
            summary = book.get('summary', 'No summary available.')
            genre = book.get('genre', 'Unknown Genre')

            user_content += f"### Retrieved Book {i + 1}:\n"
            user_content += f"Title: {title}\n"
            user_content += f"Author: {author}\n"
            user_content += f"Genre: {genre}\n"
            user_content += f"Summary: {summary}\n\n"

        user_content += (
            "Based on the user's request and the above retrieved book information, "
            "please suggest 3-5 *new* relevant book recommendations. "
            "For each recommendation, provide the title and author, and a brief, compelling reason why it matches the user's request or the themes of the retrieved books. "
            "Present the recommendations in a clear, easy-to-read list format.\n"
        )
    else:
        user_content += (
            "No highly relevant books were found in the database for the query. "
            "Please still try to suggest 3-5 general books related to the user's query, "
            "or explain if the query is too niche for common recommendations.\n"
            "Present the recommendations in a clear, easy-to-read list format.\n"
        )

    messages.append({"role": "user", "content": user_content})

    print("\n--- Sending prompt to OpenAI LLM ---")
    # print(messages) # Uncomment this line if you want to see the full messages array being sent

    try:
        response = client.chat.completions.create(
            model=llm_model_name,
            messages=messages,
            temperature=0.7,  # Controls randomness. Lower for more focused, higher for more creative.
            max_tokens=500,  # Max tokens for the LLM's response
            # You can add more parameters like top_p, frequency_penalty, presence_penalty
            # if you want to fine-tune the output.
        )

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            return "OpenAI LLM did not return a valid response (no choices or content found)."
    except openai.APIError as e:
        return f"An OpenAI API error occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred during LLM generation: {e}"


# --- 4. Main Execution (putting it all together) ---

if __name__ == "__main__":
    print("--- Starting RAG System (OpenAI Version) ---")

    # 1. Prepare the vector database (embedding and FAISS indexing)
    faiss_index, _, embedding_model = prepare_vector_database(book_data)

    print("\n--- FAISS Index and Embedding Model Ready ---")

    # 2. Define a dummy user query
    user_query = "I'm looking for a high fantasy adventure with elves and magic."
    # user_query = "Suggest a dark fantasy book with political intrigue."
    # user_query = "Tell me about books on space exploration."

    # 3. Retrieve relevant books using the retrieval function
    num_recommendations_from_db = 3
    retrieved_books = retrieve_books(user_query, faiss_index, book_data, embedding_model,
                                     num_results=num_recommendations_from_db)

    # 4. Generate final recommendations using the OpenAI LLM
    print("\n--- Generating OpenAI LLM Recommendations ---")
    final_recommendations = generate_recommendations_openai(user_query, retrieved_books, llm_model_name="gpt-3.5-turbo")

    print("\n--- Final Book Recommendations from RAG System (OpenAI) ---")
    print(final_recommendations)

    print("\n--- RAG System Process Complete ---")