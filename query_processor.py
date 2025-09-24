# query_processor.py

from typing import List, Dict
import config
import llm_utils
import database_utils
import vector_store_utils 
import re

def get_context_from_vector_store(query_text: str, collection_key: str, llm_app_components):
    """Helper to retrieve context from a specific vector store collection."""
    if not llm_app_components or "embedding_model" not in llm_app_components:
        print("Error: Embedding model not available in llm_app_components.")
        return ""
            
    collection_name_map = {
        "faqs": config.CHROMA_COLLECTION_FAQS,
        "policies": config.CHROMA_COLLECTION_POLICIES,
        "products": config.CHROMA_COLLECTION_PRODUCTS,
    }
    if collection_key not in collection_name_map:
        print(f"Warning: Invalid collection key '{collection_key}' for context retrieval.")
        return ""

    try:
        retriever = vector_store_utils.get_langchain_chroma_retriever(
            collection_name=collection_name_map[collection_key],
            embedding_model=llm_app_components["embedding_model"],
            k_results=3
        )
        retrieved_docs = retriever.invoke(query_text)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return context
    except Exception as e:
        print(f"Error retrieving context from '{collection_key}': {e}")
        return ""

def query_policy_context(query_text: str, policy_type: str, llm_app_components) -> str:
    """Queries the policy vector store for relevant context."""
    return get_context_from_vector_store(query_text, "policies", llm_app_components)


def query_faq_context(query_text: str, llm_app_components) -> str:
    """Queries the FAQ vector store for relevant context."""
    return get_context_from_vector_store(query_text, "faqs", llm_app_components)


def process_query(
    query_text: str,
    conversation_history: List[Dict[str, str]],
    llm_app_components: dict
) -> str:
    """
    Processes the user query: classifies intent, retrieves context, generates and refines response.
    """
    if not llm_app_components or "llm_chains" not in llm_app_components:
        return "I'm sorry, the system is not fully initialized. Please try again later."
    
    llm_chains = llm_app_components["llm_chains"]
    
    print(f"Processing query: '{query_text}' with history length: {len(conversation_history)}")

    intent = llm_utils.classify_intent_with_llm(query_text, llm_chains)
    if intent not in ["product_query", "return_policy", "shipping_policy", "privacy_policy", "general_faq", "greeting"]:
        intent = llm_utils.classify_intent_rules(query_text)
    print(f"Final classified intent: {intent}")

    draft_response = ""
    chain_input = { "question": query_text, "history": conversation_history, "context": "", "db_query_results": "" }

    try:
        if intent == "greeting":
            return "Hello! VASUKI is a premier jewelry company. How can I assist you with our collections or services today?"

        elif intent == "product_query":
            rewritten_query = query_text
            if conversation_history and not re.search(r'([A-Z]{2,}\d{4,})', query_text, re.IGNORECASE):
                history_str_parts = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation_history]
                rewrite_input = {"question": query_text, "history_string": "\n".join(history_str_parts), "history": []}
                rewritten_query = llm_chains["query_rewriter"].invoke(rewrite_input).strip()
                print(f"Original query: '{query_text}'. Rewritten query for search: '{rewritten_query}'")
            
            search_query = rewritten_query
            
            # --- EXTRACT PRICE LIMIT FROM QUERY ---
            price_limit = None
            price_match = re.search(r"(under|less than|below|around)\s*(\d+)", search_query, re.IGNORECASE)
            if price_match:
                price_limit = int(price_match.group(2))
                print(f"Price limit detected: {price_limit}")

            sku_match = re.search(r'([A-Z]{2,}\d{4,})', search_query, re.IGNORECASE)
            db_results = ""

            if sku_match:
                sku = sku_match.group(1).upper()
                print(f"Direct SKU lookup detected for: {sku}")
                db_results = database_utils.get_products_by_skus([sku])
            
            if not db_results:
                if sku_match: print(f"SKU '{sku_match.group(1).upper()}' not found, falling back to semantic search.")
                
                retrieved_docs = vector_store_utils.get_langchain_chroma_retriever(
                    collection_name=config.CHROMA_COLLECTION_PRODUCTS,
                    embedding_model=llm_app_components["embedding_model"], k_results=15 # Retrieve more to allow for filtering
                ).invoke(search_query)
                
                skus_from_vector = [doc.metadata.get("sku") for doc in retrieved_docs if doc.metadata.get("sku")]
                if skus_from_vector:
                    # --- PASS PRICE LIMIT TO DATABASE FUNCTION ---
                    db_results = database_utils.get_products_by_skus(skus_from_vector, price_limit=price_limit)

            if not db_results:
                draft_response = "I'm sorry, I couldn't find any products that match your description. Can I help with anything else?"
            else:
                draft_response = db_results

        elif intent.endswith("_policy"): 
            policy_type_key = intent.split('_')[0] 
            chain_input["context"] = query_policy_context(query_text, policy_type_key, llm_app_components)
            draft_response = llm_chains["policy"].invoke(chain_input)
        
        elif intent == "general_faq":
            if "what is vasuki" in query_text.lower():
                draft_response = "VASUKI is a distinguished jewelry company, known for its exquisite craftsmanship and unique designs."
            else:
                chain_input["context"] = query_faq_context(query_text, llm_app_components)
                draft_response = llm_chains["faq"].invoke(chain_input)
        
        else:
            draft_response = "I'm sorry, I'm not sure how to help with that. Can I assist with a product search or a policy question?"

        final_response = draft_response

    except Exception as e:
        print(f"Error during query processing pipeline for query '{query_text}': {e}")
        final_response = "I'm sorry, I encountered an unexpected issue while processing your request. Please try again."

    if intent == "product_query" and ("SKU:" in final_response or "Price: ₹" in final_response):
        final_response += "\n\nWould you like more details on any of these items, or can I help you find something different?"

    final_response = re.sub(r'\*\*(.*?)\*\*', r'\1', final_response).strip()
    final_response = re.sub(r'[ \t]{2,}', ' ', final_response)

    return final_response.strip()