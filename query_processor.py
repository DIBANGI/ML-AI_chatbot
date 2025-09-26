from typing import List, Dict
import config
import llm_utils
import database_utils
import vector_store_utils 
import re
import random

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

def format_product_recommendations(products: List[Dict]) -> str:
    """Formats a list of product dictionaries into a structured string."""
    if not products:
        return ""
    product_strings = []
    for product in products:
        product_str = (
            f"SKU: {product['sku']}\n"
            f"Name: {product['name']}\n"
            f"Price: ₹{product['price']}\n"
            f"Description: {product['description']}"
        )
        product_strings.append(product_str)
    return "\n\n---\n\n".join(product_strings)

def process_query(
    query_text: str,
    session_id: str,
    conversation_history: List[Dict[str, str]],
    llm_app_components: dict
) -> str:
    """
    Processes the user query: handles greetings, paginates products, classifies intent, and generates responses.
    """
    query_lower = query_text.lower().strip()

    # --- Handle "show more" for products ---
    if query_lower in ["yes", "yes please", "sure", "ok", "yep", "show more"]:
        session_recs = llm_app_components["product_recommendations"].get(session_id)
        if session_recs and session_recs["products"]:
            current_index = session_recs["index"]
            remaining_products = session_recs["products"][current_index:]
            
            if not remaining_products:
                return "You've seen all the recommendations I have for your last search. You can ask me to find something else!"

            next_batch = remaining_products[:3]
            session_recs["index"] += len(next_batch)
            
            formatted_products = format_product_recommendations(next_batch)
            response = f"Of course, here are the next few items:\n\n{formatted_products}"
            
            # Check if there are more products to show after this batch
            if len(session_recs["products"]) > session_recs["index"]:
                response += "\n\nWould you like to see even more?"
            else:
                response += "\n\nThat's all I have for this search. For more options, please visit our website at [shopvasuki.com](http://shopvasuki.com) or contact us at +91-1234567890."
            return response

    # --- Handle Greetings ---
    if "how are you" in query_lower:
        return random.choice([
            "I'm doing great, thank you! How can I assist you today?",
            "I'm just a bot, but I'm ready to help! What can I do for you?",
            "Feeling helpful! Thanks for asking. What can I get for you?"
        ])

    greetings = {
        "good morning": ["Good morning! How can I help you today?", "A very good morning to you! What are you looking for today?"],
        "good evening": ["Good evening! How may I assist you?", "Good evening! I hope you're having a pleasant evening. How can I help?"],
        "namaste": ["Namaste! How can I help you today?", "Namaste! Welcome to Vasuki. What can I do for you?"],
        "hello": ["Hello! This is Vasuki, your jewelry assistant. How can I help?", "Hi there! I'm Vasuki. Ask me anything about our products."],
        "hi": ["Hi! I'm Vasuki, ready to help with your E-commerce questions.", "Hey! Vasuki here. What can I do for you?"],
        "hey": ["Hey there! How can I assist you?", "Hey! I'm Vasuki. Let me know what you need."]
    }

    for trigger, responses in greetings.items():
        if trigger in query_lower:
            return random.choice(responses)

    if not llm_app_components or "llm_chains" not in llm_app_components:
        return "I'm sorry, the system is not fully initialized. Please try again later."
    
    # --- Reset recommendations on new query ---
    llm_app_components["product_recommendations"][session_id] = {"products": [], "index": 0}

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
            return random.choice([
                "Hello! VASUKI is a premier jewelry company. How can I assist you?",
                "Greetings! Welcome to VASUKI. How may I help you?"
            ])

        elif intent == "product_query":
            # ... (Existing product query logic to get SKUs)
            rewritten_query = query_text
            if conversation_history and not re.search(r'([A-Z]{2,}\d{4,})', query_text, re.IGNORECASE):
                history_str_parts = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation_history]
                rewrite_input = {"question": query_text, "history_string": "\n".join(history_str_parts), "history": []}
                rewritten_query = llm_chains["query_rewriter"].invoke(rewrite_input).strip()
                print(f"Original query: '{query_text}'. Rewritten query for search: '{rewritten_query}'")
            
            search_query = rewritten_query
            price_limit = None
            price_match = re.search(r"(under|less than|below|around)\s*(\d+)", search_query, re.IGNORECASE)
            if price_match:
                price_limit = int(price_match.group(2))

            sku_match = re.search(r'([A-Z]{2,}\d{4,})', search_query, re.IGNORECASE)
            db_results_list = []

            if sku_match:
                sku = sku_match.group(1).upper()
                db_results_list = database_utils.get_products_by_skus([sku])
            
            if not db_results_list:
                retrieved_docs = vector_store_utils.get_langchain_chroma_retriever(
                    collection_name=config.CHROMA_COLLECTION_PRODUCTS,
                    embedding_model=llm_app_components["embedding_model"], k_results=15
                ).invoke(search_query)
                
                skus_from_vector = [doc.metadata.get("sku") for doc in retrieved_docs if doc.metadata.get("sku")]
                if skus_from_vector:
                    db_results_list = database_utils.get_products_by_skus(skus_from_vector, price_limit=price_limit)

            # --- New Product Pagination Logic ---
            if not db_results_list:
                draft_response = "I'm sorry, I couldn't find any products that match your description. Can I help with anything else?"
            else:
                # Store all fetched products in the session
                session_recs = llm_app_components["product_recommendations"][session_id]
                session_recs["products"] = db_results_list
                
                # Get the first batch of 3
                first_batch = session_recs["products"][:3]
                session_recs["index"] = len(first_batch)
                
                formatted_products = format_product_recommendations(first_batch)
                draft_response = f"I found a few items for you:\n\n{formatted_products}"

                if len(session_recs["products"]) > 3:
                    draft_response += "\n\nWould you like me to recommend any more products?"
                else:
                    draft_response += "\n\nFor more options, please visit our website at [shopvasuki.com](http://shopvasuki.com) or contact us at +91-1234567890."

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

    # Final cleanup of the response string
    final_response = re.sub(r'\*\*(.*?)\*\*', r'\1', final_response).strip()
    final_response = re.sub(r'[ \t]{2,}', ' ', final_response)

    return final_response.strip()