# vector_store_utils.py

import os
import pandas as pd
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma # Corrected import

import config
import database_utils

def init_chroma_client():
    """Initializes and returns a persistent ChromaDB client."""
    os.makedirs(config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIRECTORY)
    return client

def get_chroma_collections(client: chromadb.PersistentClient, embedding_func):
    """Gets or creates ChromaDB collections."""
    collections = {}
    collection_names_map = {
        "faqs": config.CHROMA_COLLECTION_FAQS,
        "policies": config.CHROMA_COLLECTION_POLICIES,
        "products": config.CHROMA_COLLECTION_PRODUCTS,
    }
    for name_key, collection_name_val in collection_names_map.items():
        try:
            collection = client.get_or_create_collection(name=collection_name_val, embedding_function=embedding_func)
            print(f"Retrieved/Created collection: {collection_name_val}")
        except Exception as e:
            print(f"Error getting/creating collection {collection_name_val}: {e}")
            # As a fallback, try creating it directly if get_or_create fails in some versions
            collection = client.create_collection(name=collection_name_val, embedding_function=embedding_func, get_or_create=True)
        collections[name_key] = collection
    return collections


def load_documents_for_vector_store():
    """Loads and processes documents from files and database for ChromaDB embedding."""
    # --- Load FAQs ---
    faq_docs, faq_ids, faq_metadata = [], [], []
    try:
        faqs_df = pd.read_csv(config.FAQS_FILE_PATH)
        print(f"Successfully loaded {config.FAQS_FILE_PATH} with {len(faqs_df)} questions")
        for idx, row in faqs_df.iterrows():
            if pd.isna(row['Question']) or pd.isna(row['Answer']): continue
            question, answer = str(row['Question']).strip(), str(row['Answer']).strip()
            if not question or not answer: continue
            
            faq_docs.append(f"Question: {question}\nAnswer: {answer}")
            faq_ids.append(f"faq_{idx}_main")
            faq_metadata.append({"type": "faq", "question": question, "source": config.FAQS_FILE_PATH})

            for i in range(1, 4):
                var_col = f"Variation {i}"
                if var_col in row and pd.notna(row[var_col]):
                    variation = str(row[var_col]).strip()
                    if variation:
                        faq_docs.append(f"Question: {variation}\nAnswer: {answer}")
                        faq_ids.append(f"faq_{idx}_var{i}")
                        faq_metadata.append({"type": "faq", "question": variation, "source": config.FAQS_FILE_PATH})
    except Exception as e:
        print(f"Error loading FAQs from {config.FAQS_FILE_PATH}: {e}")

    # --- Load Policy Documents ---
    policy_docs, policy_ids, policy_metadata = [], [], []
    policy_files_map = {
        "privacy": config.PRIVACY_POLICY_FILE_PATH,
        "return": config.RETURN_POLICY_FILE_PATH,
        "shipping": config.SHIPPING_POLICY_FILE_PATH,
    }
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    for policy_type, file_path in policy_files_map.items():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"Successfully loaded {file_path} with {len(content)} characters")
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                policy_docs.append(chunk)
                policy_ids.append(f"{policy_type}_{i}")
                policy_metadata.append({"type": "policy", "policy_type": policy_type, "source": file_path, "chunk": i})
        except Exception as e:
            print(f"Error loading policy file {file_path}: {e}")

    # --- Load Product Information from SQL Database ---
    product_docs, product_ids, product_metadata = [], [], []
    conn = database_utils.get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT i.`SKU Number`, c.category_name, c.subcategory_name,
                       s.stone_name, cl.color_name, f.finish_name,
                       i.`Weight`, i.length, i.width, p.`Final SP` as unit_price,
                       i.`Status`, i.`Year of Purchase`
                FROM inventory_items i
                LEFT JOIN categories c ON i.category_id = c.id
                LEFT JOIN stones s ON i.stone_id = s.id
                LEFT JOIN colors cl ON i.color_id = cl.id
                LEFT JOIN finishes f ON i.finish_id = f.id
                LEFT JOIN pricing p ON i.`SKU Number` = p.`SKU Number`
            """)
            products_from_db = cursor.fetchall()
            print(f"Loaded {len(products_from_db)} products from database for embedding.")
            for prod_idx, product in enumerate(products_from_db):
                clean_product = {k: (v if v is not None else "") for k, v in dict(product).items()}
                
                product_text_parts = [
                    f"Product: {clean_product['category_name']} {clean_product['subcategory_name']}".strip(),
                    f"SKU: {clean_product['SKU Number']}"
                ]
                if clean_product.get('stone_name'): product_text_parts.append(f"Stone: {clean_product['stone_name']}")
                
                price = float(clean_product.get('unit_price', 0.0) or 0.0)
                product_text_parts.append(f"Price: ₹{price:.2f}")
                
                product_text = "\n".join(filter(None, product_text_parts))
                product_docs.append(product_text)
                product_ids.append(f"product_{clean_product['SKU Number']}_{prod_idx}")
                product_metadata.append({
                    "type": "product", "sku": str(clean_product['SKU Number']),
                    "category": clean_product.get('category_name', ""),
                    "price": price, "source": "database"
                })
        except Exception as e:
            print(f"Error loading product data from SQL for embedding: {e}")
        finally:
            if conn: conn.close()
    
    return {
        "faqs": {"documents": faq_docs, "ids": faq_ids, "metadatas": faq_metadata},
        "policies": {"documents": policy_docs, "ids": policy_ids, "metadatas": policy_metadata},
        "products": {"documents": product_docs, "ids": product_ids, "metadatas": product_metadata}
    }

def validate_metadata(metadata_list: list) -> list:
    """Ensures all metadata values are valid types for ChromaDB."""
    valid_metadata_list = []
    for metadata_dict in metadata_list:
        validated_item = {}
        if isinstance(metadata_dict, dict):
            for key, value in metadata_dict.items():
                if isinstance(value, (str, int, float, bool)):
                    validated_item[key] = value
                elif value is None:
                    validated_item[key] = ""
                else:
                    validated_item[key] = str(value)
        valid_metadata_list.append(validated_item)
    return valid_metadata_list

def add_documents_in_batches(collection, documents: list, ids: list, metadatas: list, batch_size=100):
    """Adds documents to a ChromaDB collection in batches."""
    if not all(isinstance(lst, list) for lst in [documents, ids, metadatas]): return False
    if not (len(documents) == len(ids) == len(metadatas)): return False
    if not documents: return True

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_meta_raw = metadatas[i:i + batch_size]
        batch_meta_validated = validate_metadata(batch_meta_raw)

        try:
            collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_meta_validated)
            print(f"Added batch {i//batch_size + 1} to '{collection.name}'.")
        except Exception as e:
            print(f"Error adding batch {i//batch_size + 1} to '{collection.name}': {e}")
            continue
    return True

def safely_clear_collection(collection):
    """Safely clears all documents from a ChromaDB collection."""
    try:
        count = collection.count()
        if count == 0: return True
        all_items = collection.get(include=[])
        if all_items and all_items['ids']:
            collection.delete(ids=all_items['ids'])
        return collection.count() == 0
    except Exception as e:
        print(f"Error clearing collection '{collection.name}': {e}")
        return False

def get_langchain_chroma_retriever(collection_name: str, embedding_model, k_results=3):
    """Creates a Langchain Chroma retriever for a given collection."""
    vectorstore = Chroma(
        client=init_chroma_client(),
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY
    )
    return vectorstore.as_retriever(search_kwargs={"k": k_results})