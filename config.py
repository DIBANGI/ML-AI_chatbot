# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# --- Model Configuration ---
LLM_MODEL_ID = "llama-3.1-8b-instant"  # Changed to Groq Llama 3 model
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# --- Groq API Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Database Configuration ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "hp09876")
DB_NAME = os.getenv("DB_NAME", "vasuki_inventory")

# --- ChromaDB Configuration ---
CHROMA_PERSIST_DIRECTORY = "./chroma_db_llama"
CHROMA_COLLECTION_FAQS = "faqs_llama"
CHROMA_COLLECTION_POLICIES = "policies_llama"
CHROMA_COLLECTION_PRODUCTS = "products_llama"

# --- File Paths ---
FAQS_FILE_PATH = "FAQs.csv"
PRIVACY_POLICY_FILE_PATH = "privacy_policy.txt"
RETURN_POLICY_FILE_PATH = "return_policy.txt"
SHIPPING_POLICY_FILE_PATH = "shipping_policy.txt"
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"

# --- Database Schema Information (for LLM context) ---
DB_SCHEMA_INFO = """
Database Schema Information:
- Main tables: inventory_items, pricing, categories, stones, colors, finishes
- Inventory table fields: SL, SKU Number, Weight, Length, Width, Year of Purchase, Status, CUSTOMER NAME, SALE AMOUNT, DOP
- Categories table: category_name and subcategory_name fields
- Reference tables: stones (stone_name), colors (color_name), finishes (finish_name)
- Pricing table fields: Unit Price, Cost price, Thread work, GST on Cost price, Packaging cost, Final Cost price, SP - Margin, Taxes, SP, Final SP

Available Fields for Queries:
- SKU Number (unique identifier)
- Category and subcategory information
- Product attributes: Weight, Length, Width, Stone, Color, Finish
- Purchase details: Year of Purchase, DOP (Date of Purchase)
- Sale information: Status, CUSTOMER NAME, SALE AMOUNT
- Pricing components: All cost and price fields including taxes and margins
"""

# --- LLM Parameters ---
# These parameters are now set directly in the Groq client, but we keep them here for reference
LLM_MAX_NEW_TOKENS = 1024  # Increased for potentially more detailed responses from Llama 3
LLM_TEMPERATURE = 0.3
# The following are less commonly used with the Groq API but kept for potential future use
LLM_TOP_P = 0.9
LLM_REPETITION_PENALTY = 1.2
LLM_DO_SAMPLE = True


# --- Policy Types ---
POLICY_TYPES = {
    "return_policy": "return",
    "shipping_policy": "shipping",
    "privacy_policy": "privacy"
}