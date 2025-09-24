# import_data.py

import pandas as pd
import sqlite3
import re
import numpy as np
import os

DB_FILE = "vasuki_inventory.db"

# --- Clean up old database file if it exists ---
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)
    print(f"Removed old database file: {DB_FILE}")

def clean_float(value):
    if pd.isna(value) or value in ['', 'NA', 'N/A', 'nan', 'NaN']:
        return None
    try:
        cleaned = re.sub(r'[^\d.-]', '', str(value))
        if cleaned in ['', '.', '-']:
            return None
        return float(cleaned)
    except (ValueError, TypeError):
        return None

def get_or_create_id(cursor, table, column, value):
    if pd.isna(value) or str(value).strip().lower() in ['', 'n/a', 'nan', 'none']:
        return None
    value = str(value).strip()
    try:
        cursor.execute(f"SELECT id FROM {table} WHERE {column} = ?", (value,))
        result = cursor.fetchone()
        if result:
            return result[0]
        cursor.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (value,))
        return cursor.lastrowid
    except Exception as e:
        print(f"Error in get_or_create_id for {table}: {e}")
        return None

def get_or_create_category_id(cursor, category, subcategory):
    if pd.isna(category) or str(category).strip() in ['', 'NA', 'N/A', 'nan']:
        return None
    subcategory = str(subcategory).strip() if not pd.isna(subcategory) else None
    if subcategory in ['', 'NA', 'N/A', 'nan']:
        subcategory = None
    
    # Handle NULL subcategory correctly in the query
    if subcategory is None:
        cursor.execute("SELECT id FROM categories WHERE category_name = ? AND subcategory_name IS NULL", (str(category).strip(),))
    else:
        cursor.execute("SELECT id FROM categories WHERE category_name = ? AND subcategory_name = ?", (str(category).strip(), subcategory))
    
    result = cursor.fetchone()
    if result:
        return result[0]
    
    cursor.execute("INSERT INTO categories (category_name, subcategory_name) VALUES (?, ?)", (str(category).strip(), subcategory))
    return cursor.lastrowid

def parse_dimensions(dim_str):
    if pd.isna(dim_str) or str(dim_str).strip() in ['', '0', 'NA', 'N/A']:
        return None, None
    try:
        parts = re.split(r'[x×]', str(dim_str).replace(' ', ''))
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            return clean_float(parts[0]), clean_float(parts[1])
        elif len(parts) == 1:
            return clean_float(parts[0]), None
        return None, None
    except Exception:
        return None, None
        
def parse_date(date_str):
    if pd.isna(date_str) or str(date_str).strip() in ['', 'NA', 'N/A']:
        return None
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y.%m.%d', '%d-%m-%Y', '%m-%d-%Y'):
        try:
            return pd.to_datetime(date_str, format=fmt).strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            continue
    return None

# --- Main Script ---
df = pd.read_csv('cleaned_inventory.csv')
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str)
df = df.replace(['nan', 'NaN', 'N/A', 'NA', 'null', 'None', ''], np.nan)

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
print(f"Successfully connected to the SQLite database: {DB_FILE}")

# --- Create Tables (SQLite syntax) ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL,
    subcategory_name TEXT,
    UNIQUE (category_name, subcategory_name)
)""")
cursor.execute("CREATE TABLE IF NOT EXISTS stones (id INTEGER PRIMARY KEY AUTOINCREMENT, stone_name TEXT NOT NULL UNIQUE)")
cursor.execute("CREATE TABLE IF NOT EXISTS colors (id INTEGER PRIMARY KEY AUTOINCREMENT, color_name TEXT NOT NULL UNIQUE)")
cursor.execute("CREATE TABLE IF NOT EXISTS finishes (id INTEGER PRIMARY KEY AUTOINCREMENT, finish_name TEXT NOT NULL UNIQUE)")

cursor.execute("""
CREATE TABLE IF NOT EXISTS inventory_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    `SL` TEXT,
    `SKU Number` TEXT UNIQUE NOT NULL,
    category_id INTEGER,
    `Weight` REAL,
    length REAL,
    width REAL,
    stone_id INTEGER,
    color_id INTEGER,
    finish_id INTEGER,
    `Year of Purchase` INTEGER,
    `Status` TEXT DEFAULT 'In Stock',
    `CUSTOMER NAME` TEXT,
    `SALE AMOUNT` REAL,
    `DOP` TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories(id),
    FOREIGN KEY (stone_id) REFERENCES stones(id),
    FOREIGN KEY (color_id) REFERENCES colors(id),
    FOREIGN KEY (finish_id) REFERENCES finishes(id)
)""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS pricing (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inventory_item_id INTEGER NOT NULL,
    `SKU Number` TEXT NOT NULL,
    `Unit Price` REAL,
    `Cost price` REAL,
    `Thread work` REAL,
    `GST on Cost price` REAL,
    `Packaging cost` REAL,
    `Final Cost price` REAL,
    `SP - Margin` TEXT,
    `Taxes` REAL,
    `SP` REAL,
    `Final SP` REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (inventory_item_id) REFERENCES inventory_items(id),
    FOREIGN KEY (`SKU Number`) REFERENCES inventory_items(`SKU Number`)
)""")

conn.commit()

# --- Import Data ---
for index, row in df.iterrows():
    try:
        sku = str(row.get('SKU Number', '')).strip().upper() if pd.notna(row.get('SKU Number')) else None
        if not sku:
            continue
            
        category_id = get_or_create_category_id(cursor, row.get('Category'), row.get('Subcategory'))
        stone_id = get_or_create_id(cursor, 'stones', 'stone_name', row.get('Stones'))
        color_id = get_or_create_id(cursor, 'colors', 'color_name', row.get('Color'))
        finish_id = get_or_create_id(cursor, 'finishes', 'finish_name', row.get('Finish'))
        length, width = parse_dimensions(row.get('Dimensions'))
        dop = parse_date(row.get('DOP'))

        cursor.execute("""
            INSERT INTO inventory_items (
                `SL`, `SKU Number`, category_id, `Weight`, length, width,
                stone_id, color_id, finish_id, `Year of Purchase`,
                `Status`, `CUSTOMER NAME`, `SALE AMOUNT`, `DOP`
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(row.get('SL', '')) if pd.notna(row.get('SL')) else None, 
            sku, category_id, clean_float(row.get('Weight')), length, width,
            stone_id, color_id, finish_id, 
            int(float(row.get('Year of Purchase'))) if pd.notna(row.get('Year of Purchase')) else None,
            str(row.get('Status', 'In Stock')).strip() or 'In Stock',
            str(row.get('CUSTOMER NAME', '')) if pd.notna(row.get('CUSTOMER NAME')) else None,
            clean_float(row.get('SALE AMOUNT')), dop
        ))
        inventory_item_id = cursor.lastrowid
        
        cursor.execute("""
            INSERT INTO pricing (
                inventory_item_id, `SKU Number`, `Unit Price`, `Cost price`,
                `Thread work`, `GST on Cost price`, `Packaging cost`,
                `Final Cost price`, `SP - Margin`, `Taxes`, `SP`, `Final SP`
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            inventory_item_id, sku, clean_float(row.get('Unit Price')), clean_float(row.get('Cost price')),
            clean_float(row.get('Thread work')), clean_float(row.get('GST on Cost price')), clean_float(row.get('Packaging cost')),
            clean_float(row.get('Final Cost price')), str(row.get('SP - Margin', '')) if pd.notna(row.get('SP - Margin')) else None,
            clean_float(row.get('Taxes')), clean_float(row.get('SP')), clean_float(row.get('Final SP'))
        ))
        
    except Exception as e:
        print(f"Error importing row {index} (SKU: {row.get('SKU Number', 'UNKNOWN')}): {e}")

conn.commit()
conn.close()
print(f"✅ Data import to {DB_FILE} complete.")