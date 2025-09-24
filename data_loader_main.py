# data_loader_main.py

import config
import vector_store_utils as vsu
from chromadb.utils import embedding_functions # For SentenceTransformerEmbeddingFunction

def main():
    """
    Standalone script to initialize ChromaDB, load all documents,
    and store their embeddings. This should be run once or when data sources change.
    """
    print("Starting ChromaDB data loading process...")

    # 1. Initialize ChromaDB Client and Embedding Function
    try:
        chroma_client = vsu.init_chroma_client()
        # Using the same SentenceTransformer model as specified in original code for ChromaDB
        # Langchain's HuggingFaceEmbeddings is for the retriever part, Chroma can use its own.
        # However, for consistency, it's often better to use the same embedding model everywhere.
        # Let's use the Langchain HuggingFaceEmbeddings wrapper for ChromaDB population too.
        # ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL_ID)
        # OR, to align with llm_utils.get_embedding_model() for Langchain retrievers:
        # This requires Langchain to be installed.
        from langchain_community.embeddings import HuggingFaceEmbeddings
        langchain_ef_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_ID)
        
        # ChromaDB's add method expects raw documents, not Langchain Document objects if using client directly.
        # If using Langchain's Chroma.from_documents, then it handles Langchain Document objects.
        # For this script, we're populating Chroma directly using its client.
        # The `embedding_function` for `client.get_or_create_collection` needs to be a Chroma-compatible one.
        chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL_ID)

        print("ChromaDB client and embedding function initialized.")
    except Exception as e:
        print(f"Error initializing ChromaDB client or embedding function: {e}")
        return

    # 2. Get or Create Collections
    try:
        collections = vsu.get_chroma_collections(chroma_client, chroma_ef)
        print(f"Collections retrieved/created: {list(collections.keys())}")
    except Exception as e:
        print(f"Error getting/creating ChromaDB collections: {e}")
        return

    # 3. Load Documents from sources
    print("Loading documents from files and database...")
    all_document_data = vsu.load_documents_for_vector_store()
    if not all_document_data:
        print("No documents were loaded. Exiting.")
        return
    print("Documents loaded successfully.")

    # 4. Clear existing data and add new documents to collections
    for collection_key, data_dict in all_document_data.items():
        if collection_key not in collections:
            print(f"Warning: No collection found for key '{collection_key}'. Skipping.")
            continue
        
        current_collection = collections[collection_key]
        print(f"\nProcessing collection: {current_collection.name} (for key '{collection_key}')")

        # Clear existing documents
        print(f"Clearing existing documents from {current_collection.name}...")
        if vsu.safely_clear_collection(current_collection):
            print(f"Successfully cleared {current_collection.name}.")
        else:
            print(f"Warning: Could not fully clear {current_collection.name}. Proceeding with adding new data.")

        # Add new documents
        docs = data_dict.get("documents")
        ids = data_dict.get("ids")
        metadatas = data_dict.get("metadatas")

        if not all([docs, ids, metadatas]):
            print(f"Missing documents, ids, or metadatas for {collection_key}. Skipping add operation.")
            continue
        
        if not (len(docs) == len(ids) == len(metadatas)):
            print(f"Data length mismatch for {collection_key}: Docs({len(docs)}), IDs({len(ids)}), Metas({len(metadatas)}). Skipping.")
            continue

        if not docs: # Check if docs list is empty
            print(f"No documents to add for {collection_key}. Skipping add operation.")
            continue

        print(f"Adding {len(docs)} documents to {current_collection.name}...")
        # Metadata validation is now inside add_documents_in_batches
        success = vsu.add_documents_in_batches(
            current_collection,
            docs,
            ids,
            metadatas, # Raw metadatas, validation happens inside
            batch_size=50 # Smaller batch size for potentially large documents/embeddings
        )
        if success:
            print(f"Successfully added documents to {current_collection.name}.")
            print(f"Collection {current_collection.name} now contains {current_collection.count()} documents.")
        else:
            print(f"Failed to add all documents to {current_collection.name}.")

    print("\nChromaDB data loading process completed.")
    print(f"Vector store data is persisted in: {config.CHROMA_PERSIST_DIRECTORY}")

if __name__ == "__main__":
    main()
