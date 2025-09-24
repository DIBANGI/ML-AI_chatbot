# llm_utils.py

import os
import re
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict

import config
import vector_store_utils

# --- LLM and Embeddings Initialization ---
def get_embedding_model():
    """Initializes and returns the HuggingFace embedding model."""
    print(f"Initializing embedding model: {config.EMBEDDING_MODEL_ID}")
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_ID)

def get_groq_chat_model():
    """Initializes and returns the ChatGroq model."""
    print(f"Initializing Groq LLM: {config.LLM_MODEL_ID}")
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    
    return ChatGroq(
        temperature=config.LLM_TEMPERATURE,
        model_name=config.LLM_MODEL_ID,
        groq_api_key=config.GROQ_API_KEY,
        max_tokens=config.LLM_MAX_NEW_TOKENS,
    )

# --- Prompt Templates ---
DEFAULT_SYSTEM_PROMPT = """You are an AI customer service representative for VASUKI Jewelry Store.
Your role is to help customers with their queries in a friendly, professional, and concise manner.
Always maintain a polite and helpful tone. Only provide information that's available in the provided context or your general knowledge if no context is given.
If information is not available, politely say so and offer to help with something else.
Format prices and measurements consistently (e.g., Price: ₹XX.XX).
Do not make up information. If the answer is not in the context, say you don't know.
When answering follow-up questions, use the previous conversation turns for context.
Directly answer the user's query without any introductory phrases about your own response.
Do not include emotive expressions or actions in asterisks (e.g., *smiles*).
"""

PRODUCT_QUERY_SYSTEM_PROMPT = """You are a highly precise product assistant for VASUKI Jewelry Store.
Your ONLY task is to present product information based STRICTLY on the "Database Query Results" provided. This is your single source of truth.

**CRITICAL RULES:**
1.  **EXCLUSIVE DATA SOURCE:** You MUST ONLY use the data within the "Database Query Results" section. Do NOT use the "Context (from vector store)" to get product details like SKUs, prices, or names. The vector store context is only for understanding the user's initial request, not for generating the answer.
2.  **NO RESULTS CASE:** If the "Database Query Results" is empty, your ONLY permitted response is: "I'm sorry, I couldn't find any products that match your description. Can I help with anything else?" Do NOT say anything else.
3.  **NO FABRICATION:** You are forbidden from inventing, assuming, or hallucinating any product details, prices, availability, or SKUs. If a detail is not in the "Database Query Results", you must omit it.
4.  **PRESENTATION:**
    - Present the relevant products conversationally.
    - For each product, list its known attributes directly from the data.
    - Format prices with the Rupee symbol (e.g., Price: ₹XX.XX).
    - Answer the user's query directly, without introductory phrases like "Here is what I found...".

Database Schema for reference (This is just for your context, not a data source):
{db_schema}
"""

POLICY_SYSTEM_PROMPT = """You are a customer service specialist for VASUKI Jewelry Store, responsible for explaining our policies clearly and accurately.
Use ONLY information provided in the context and relevant parts of the conversation history to answer policy-related questions.
Structure your response in a clear, easy-to-understand format. Highlight key points.
If specific details aren't available in the context, acknowledge this and suggest they contact customer support directly.
Directly answer the user's query without any introductory phrases.
"""

FAQ_SYSTEM_PROMPT = """You are a friendly jewelry expert at VASUKI Jewelry Store.
Answer the customer's question based ONLY on the provided FAQ context and relevant conversation history.
Keep responses conversational and natural.
If the exact question isn't in the FAQ context, state that the information is not available in the FAQs and offer to help with other questions.
Directly answer the user's query.
"""

INTENT_CLASSIFICATION_SYSTEM_PROMPT = """TASK: Your single-minded purpose is to classify the user's intent.
Analyze the user's question and respond with ONLY ONE of the following category names.

RULES:
- Your response MUST be a single word from the list below.
- Do NOT add any pleasantries, explanations, or punctuation.
- Analyze ONLY the LATEST customer question. Ignore conversation history.

CATEGORIES:
- product_query
- return_policy
- shipping_policy
- privacy_policy
- general_faq
- greeting
- other
"""

REFINEMENT_SYSTEM_PROMPT = """You are a professional customer service specialist.
Refine the DRAFT RESPONSE to be conversational and customer-friendly, considering the conversation history.
The final response should directly address the user's query using ONLY the factual information present in the DRAFT RESPONSE.
DO NOT introduce any new product details, SKUs, prices, or other factual information not already in the draft.
Remove any technical language and format for easy reading. Add a polite closing if appropriate.
"""

QUERY_REWRITING_SYSTEM_PROMPT = """You are an expert at rephrasing questions. Your task is to rewrite a follow-up question from a user into a self-contained, standalone question.
Use the conversation history to understand the context.
The rewritten question should be concise and optimized for a vector database search.

Example 1:
History:
User: "Do you have any gold necklaces?"
Assistant: "Yes, we have several..."
Current Question: "what about under 10000?"
Standalone Question: "gold necklaces under 10000"

Example 2:
History:
User: "show me bangle options"
Assistant: "Here are some bangles..."
Current Question: "only the silver ones"
Standalone Question: "silver bangles"

Example 3:
History:
User: "tell me about SPSLB0004"
Assistant: "Here are some details for SPSLB0004..."
Current Question: "give me more details"
Standalone Question: "full details for SPSLB0004"

Respond ONLY with the rewritten standalone question. Do not add any other text.
"""


# --- User Templates ---
GENERAL_USER_TEMPLATE = "Context:\n{context}\n\nCurrent Question: {question}"
PRODUCT_USER_TEMPLATE = "Database Query Results:\n{db_query_results}\n\nContext (from vector store, for reference ONLY):\n{context}\n\nCurrent Question: {question}"
POLICY_USER_TEMPLATE = "Policy Context:\n{context}\n\nCurrent Question: {question}"
FAQ_USER_TEMPLATE = "FAQ Context:\n{context}\n\nCurrent Question: {question}"
INTENT_USER_TEMPLATE = "Customer Question: {question}"
REFINEMENT_USER_TEMPLATE = "Original Customer Question: {question}\nConversation History (if any):\n{history_string}\n\nDraft Response to Refine:\n{draft_response}"
QUERY_REWRITING_USER_TEMPLATE = "Conversation History:\n{history_string}\n\nFollow-up Question: {question}"


# --- Langchain Chains ---
def create_llm_chains(llm: ChatGroq, embedding_model):
    """Creates and returns a dictionary of Langchain chains."""

    def get_retriever(collection_name_key: str, k: int = 3):
        collection_name_map = {
            "faqs": config.CHROMA_COLLECTION_FAQS,
            "policies": config.CHROMA_COLLECTION_POLICIES,
            "products": config.CHROMA_COLLECTION_PRODUCTS,
        }
        if collection_name_key not in collection_name_map:
            raise ValueError(f"Invalid collection key for retriever: {collection_name_key}")
        return vector_store_utils.get_langchain_chroma_retriever(
            collection_name_map[collection_name_key], embedding_model, k_results=k
        )

    def create_chain(system_prompt_text, user_template_text, include_retriever_key=None):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_text),
            ("placeholder", "{history}"),
            ("human", user_template_text),
        ])

        def prepare_chain_input(input_dict: Dict):
            if include_retriever_key and "context" not in input_dict and "db_query_results" not in input_dict:
                retriever = get_retriever(include_retriever_key)
                retrieved_docs = retriever.invoke(input_dict["question"])
                input_dict["context"] = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            history_messages = []
            for turn in input_dict.get("history", []):
                role = turn.get("role")
                if role == "user":
                    history_messages.append(HumanMessage(content=turn["content"]))
                elif role == "assistant":
                    history_messages.append(AIMessage(content=turn["content"]))
            input_dict["history"] = history_messages
            
            if "history_string" in user_template_text:
                history_str_parts = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in input_dict.get("raw_history", [])]
                input_dict["history_string"] = "\n".join(history_str_parts)

            return input_dict

        chain = (
            RunnablePassthrough.assign(raw_history=lambda x: x.get("history", []))
            | RunnableLambda(prepare_chain_input)
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    chains = {
        "intent": create_chain(INTENT_CLASSIFICATION_SYSTEM_PROMPT, INTENT_USER_TEMPLATE),
        "general": create_chain(DEFAULT_SYSTEM_PROMPT, GENERAL_USER_TEMPLATE, include_retriever_key="faqs"),
        "product": create_chain(PRODUCT_QUERY_SYSTEM_PROMPT.format(db_schema=config.DB_SCHEMA_INFO), PRODUCT_USER_TEMPLATE),
        "policy": create_chain(POLICY_SYSTEM_PROMPT, POLICY_USER_TEMPLATE, include_retriever_key="policies"),
        "faq": create_chain(FAQ_SYSTEM_PROMPT, FAQ_USER_TEMPLATE, include_retriever_key="faqs"),
        "refinement": create_chain(REFINEMENT_SYSTEM_PROMPT, REFINEMENT_USER_TEMPLATE),
        "query_rewriter": create_chain(QUERY_REWRITING_SYSTEM_PROMPT, QUERY_REWRITING_USER_TEMPLATE)
    }
    return chains

def classify_intent_with_llm(query: str, llm_chains) -> str:
    """Classifies intent using the LLM. Based on current query ONLY."""
    if "intent" not in llm_chains:
        return "other"
    try:
        response = llm_chains["intent"].invoke({"question": query, "history": []})
        cleaned_intent = response.strip().lower().replace(".", "")
        
        valid_intents = [
            "product_query", "return_policy", "shipping_policy", 
            "privacy_policy", "general_faq", "greeting", "other"
        ]
        
        if cleaned_intent in valid_intents:
            return cleaned_intent
        
        for intent in valid_intents:
            if intent in cleaned_intent:
                return intent
                
        return "other"
    except Exception as e:
        print(f"Error during LLM intent classification: {e}")
        return "other"

def classify_intent_rules(query: str) -> str:
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["hello", "hi", "hey", "greetings"]):
        return "greeting"
    if any(kw in query_lower for kw in ["return", "refund", "exchange"]):
        return "return_policy"
    if any(kw in query_lower for kw in ["shipping", "delivery", "track"]):
        return "shipping_policy"
    if any(kw in query_lower for kw in ["privacy", "data", "personal information"]):
        return "privacy_policy"
    if any(kw in query_lower for kw in ["product", "item", "jewelry", "ring", "necklace", "price", "cost", "buy", "find"]):
        return "product_query"
    return "general_faq"