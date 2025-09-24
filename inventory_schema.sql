-- Create the database
CREATE DATABASE IF NOT EXISTS vasuki_inventory;
USE vasuki_inventory;

-- Drop existing tables if they exist
DROP TABLE IF EXISTS pricing;
DROP TABLE IF EXISTS inventory_items;
DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS stones;
DROP TABLE IF EXISTS colors;
DROP TABLE IF EXISTS finishes;
DROP VIEW IF EXISTS inventory_overview;
DROP PROCEDURE IF EXISTS add_inventory_item;

-- Create categories table
CREATE TABLE categories (
    id INT PRIMARY KEY AUTO_INCREMENT,
    category_name VARCHAR(50) NOT NULL,
    subcategory_name VARCHAR(50)
);

-- Create stones table
CREATE TABLE stones (
    id INT PRIMARY KEY AUTO_INCREMENT,
    stone_name VARCHAR(50) NOT NULL
);

-- Create colors table
CREATE TABLE colors (
    id INT PRIMARY KEY AUTO_INCREMENT,
    color_name VARCHAR(50) NOT NULL
);

-- Create finishes table
CREATE TABLE finishes (
    id INT PRIMARY KEY AUTO_INCREMENT,
    finish_name VARCHAR(50) NOT NULL
);

-- Create main inventory table
CREATE TABLE inventory_items (
    id INT PRIMARY KEY AUTO_INCREMENT,
    `SL` VARCHAR(20),
    `SKU Number` VARCHAR(20) UNIQUE NOT NULL,
    category_id INT,
    `Weight` DECIMAL(10,2),
    length DECIMAL(10,2),
    width DECIMAL(10,2),
    stone_id INT,
    color_id INT,
    finish_id INT,
    `Year of Purchase` INT,
    `Status` VARCHAR(20) DEFAULT 'In Stock',
    `CUSTOMER NAME` VARCHAR(100),
    `SALE AMOUNT` DECIMAL(10,2),
    `DOP` DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories(id),
    FOREIGN KEY (stone_id) REFERENCES stones(id),
    FOREIGN KEY (color_id) REFERENCES colors(id),
    FOREIGN KEY (finish_id) REFERENCES finishes(id)
);

-- Create pricing table
CREATE TABLE pricing (
    id INT PRIMARY KEY AUTO_INCREMENT,
    inventory_item_id INT NOT NULL,
    `SKU Number` VARCHAR(20) NOT NULL,
    `Unit Price` DECIMAL(10,2),
    `Cost price` DECIMAL(10,2),
    `Thread work` DECIMAL(10,2),
    `GST on Cost price` DECIMAL(10,2),
    `Packaging cost` DECIMAL(10,2),
    `Final Cost price` DECIMAL(10,2),
    `SP - Margin` VARCHAR(10),
    `Taxes` DECIMAL(5,2),
    `SP` DECIMAL(10,2),
    `Final SP` DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (inventory_item_id) REFERENCES inventory_items(id),
    FOREIGN KEY (`SKU Number`) REFERENCES inventory_items(`SKU Number`)
);

-- Create indexes for better query performance
CREATE INDEX idx_sku ON inventory_items(`SKU Number`);
CREATE INDEX idx_status ON inventory_items(`Status`);
CREATE INDEX idx_category ON inventory_items(category_id);



-- Create a view for inventory overview
CREATE VIEW inventory_overview AS
SELECT 
    i.id,
    i.`SL`,
    i.`SKU Number`,
    c.category_name,
    c.subcategory_name,
    i.`Weight`,
    i.length,
    i.width,
    s.stone_name,
    col.color_name,
    f.finish_name,
    i.`Year of Purchase`,
    p.`Unit Price`,
    p.`Cost price`,
    p.`Thread work`,
    p.`GST on Cost price`,
    p.`Packaging cost`,    p.`Final Cost price`,
    p.`SP - Margin`,
    p.`Taxes`,
    p.`SP`,
    p.`Final SP`,
    i.`Status`,
    i.`CUSTOMER NAME`,
    i.`SALE AMOUNT`,
    i.`DOP`
FROM inventory_items i
LEFT JOIN categories c ON i.category_id = c.id
LEFT JOIN stones s ON i.stone_id = s.id
LEFT JOIN colors col ON i.color_id = col.id
LEFT JOIN finishes f ON i.finish_id = f.id
LEFT JOIN pricing p ON i.id = p.inventory_item_id;

-- Create procedure to add new inventory item
DELIMITER //

CREATE PROCEDURE add_inventory_item(
    IN p_sl VARCHAR(20),
    IN p_sku VARCHAR(20),
    IN p_category VARCHAR(50),
    IN p_subcategory VARCHAR(50),
    IN p_weight DECIMAL(10,2),
    IN p_length DECIMAL(10,2),
    IN p_width DECIMAL(10,2),
    IN p_stone VARCHAR(50),
    IN p_color VARCHAR(50),
    IN p_finish VARCHAR(50),
    IN p_year INT,
    IN p_unit_price DECIMAL(10,2),
    IN p_thread_work DECIMAL(10,2),
    IN p_gst DECIMAL(10,2),
    IN p_packaging DECIMAL(10,2),
    IN p_sp_margin VARCHAR(10),
    IN p_taxes DECIMAL(5,2)
)
BEGIN
    DECLARE v_category_id INT;
    DECLARE v_stone_id INT;
    DECLARE v_color_id INT;
    DECLARE v_finish_id INT;
    DECLARE v_inventory_id INT;
    DECLARE v_cost_price DECIMAL(10,2);
    DECLARE v_gst DECIMAL(10,2);
    DECLARE v_final_cost DECIMAL(10,2);
    DECLARE v_selling_price DECIMAL(10,2);
    DECLARE v_final_sp DECIMAL(10,2);    DECLARE v_margin_value DECIMAL(10,2);
    
    -- Calculate all the pricing values
    SET v_cost_price = COALESCE(p_unit_price, 0);
    SET v_gst = COALESCE(p_gst, 0);
    SET v_final_cost = COALESCE(v_cost_price, 0) + COALESCE(p_thread_work, 0) + COALESCE(v_gst, 0) + COALESCE(p_packaging, 0);
    
    -- Calculate margin value from percentage
    SET v_margin_value = CAST(REPLACE(COALESCE(p_sp_margin, '0%'), '%', '') AS DECIMAL(10,2)) / 100;
    
    -- Calculate selling price and final selling price
    SET v_selling_price = v_final_cost * (1 + v_margin_value);
    SET v_final_sp = v_selling_price * (1 + COALESCE(p_taxes, 0) / 100);
    
    -- Get or create category
    SELECT id INTO v_category_id FROM categories 
    WHERE category_name = p_category AND (subcategory_name = p_subcategory OR (subcategory_name IS NULL AND p_subcategory IS NULL));
    
    IF v_category_id IS NULL THEN
        INSERT INTO categories (category_name, subcategory_name)
        VALUES (p_category, p_subcategory);
        SET v_category_id = LAST_INSERT_ID();
    END IF;
    
    -- Get or create stone
    IF p_stone IS NOT NULL AND p_stone != '' THEN
        SELECT id INTO v_stone_id FROM stones WHERE stone_name = p_stone;
        IF v_stone_id IS NULL THEN
            INSERT INTO stones (stone_name) VALUES (p_stone);
            SET v_stone_id = LAST_INSERT_ID();
        END IF;
    END IF;
    
    -- Get or create color
    IF p_color IS NOT NULL AND p_color != '' THEN
        SELECT id INTO v_color_id FROM colors WHERE color_name = p_color;
        IF v_color_id IS NULL THEN
            INSERT INTO colors (color_name) VALUES (p_color);
            SET v_color_id = LAST_INSERT_ID();
        END IF;
    END IF;
    
    -- Get or create finish
    IF p_finish IS NOT NULL AND p_finish != '' THEN
        SELECT id INTO v_finish_id FROM finishes WHERE finish_name = p_finish;
        IF v_finish_id IS NULL THEN
            INSERT INTO finishes (finish_name) VALUES (p_finish);
            SET v_finish_id = LAST_INSERT_ID();
        END IF;
    END IF;
      -- Insert inventory item
    INSERT INTO inventory_items (
        `SKU Number`, category_id, `Weight`, length, width,
        stone_id, color_id, finish_id, `Year of Purchase`
    )
    VALUES (
        p_sku, v_category_id, p_weight, p_length, p_width,
        v_stone_id, v_color_id, v_finish_id, p_year
    );
    
    SET v_inventory_id = LAST_INSERT_ID();
    
    -- Calculate prices and Insert pricing information
    INSERT INTO pricing (
        inventory_item_id, `SKU Number`, `Unit Price`, `Cost price`,
        `Thread work`, `GST on Cost price`, `Packaging cost`,
        `Final Cost price`, `SP - Margin`, `Taxes`, `SP`, `Final SP`
    )
    VALUES (
        v_inventory_id,
        p_sku,
        p_unit_price, 
        v_cost_price,
        p_thread_work, 
        v_gst, 
        p_packaging,
        v_final_cost,
        p_sp_margin,
        p_taxes,
        v_selling_price,
        v_final_sp
    );
END //

DELIMITER ;

-- Create a stored procedure to import CSV data
DELIMITER //

CREATE PROCEDURE import_inventory_data()
BEGIN
    -- Add code here to import data from CSV
    -- This would typically be done through your application layer
    -- Or using LOAD DATA INFILE in MySQL
END //

-- Create utility function to parse date
DELIMITER //

CREATE FUNCTION parse_date(date_str VARCHAR(50))
RETURNS DATE
DETERMINISTIC
BEGIN
    DECLARE result DATE;
    
    -- Try different date formats
    SET result = STR_TO_DATE(date_str, '%m/%d/%Y');
    IF result IS NULL THEN
        SET result = STR_TO_DATE(date_str, '%d/%m/%Y');
    END IF;
    IF result IS NULL THEN
        SET result = STR_TO_DATE(date_str, '%Y-%m-%d');
    END IF;
    
    RETURN result;
END //

-- Create procedure to import a single inventory item
CREATE PROCEDURE import_inventory_item(
    IN p_sl VARCHAR(20),
    IN p_sku VARCHAR(20),
    IN p_category VARCHAR(50),
    IN p_subcategory VARCHAR(50),
    IN p_weight DECIMAL(10,2),
    IN p_length DECIMAL(10,2),
    IN p_width DECIMAL(10,2),
    IN p_stone VARCHAR(50),
    IN p_color VARCHAR(50),
    IN p_finish VARCHAR(50),
    IN p_year INT,
    IN p_unit_price DECIMAL(10,2),
    IN p_cost_price DECIMAL(10,2),
    IN p_thread_work DECIMAL(10,2),
    IN p_gst DECIMAL(10,2),
    IN p_packaging DECIMAL(10,2),
    IN p_final_cost DECIMAL(10,2),
    IN p_sp_margin VARCHAR(10),
    IN p_taxes DECIMAL(5,2),
    IN p_selling_price DECIMAL(10,2),
    IN p_final_sp DECIMAL(10,2),
    IN p_status VARCHAR(20),
    IN p_customer VARCHAR(100),
    IN p_sale_amount DECIMAL(10,2),
    IN p_dop VARCHAR(50)
)
BEGIN
    DECLARE v_category_id INT;
    DECLARE v_stone_id INT;
    DECLARE v_color_id INT;
    DECLARE v_finish_id INT;
    DECLARE v_inventory_id INT;
    DECLARE v_date_of_purchase DATE;
    
    -- Get date
    IF p_dop IS NOT NULL AND p_dop != '' THEN
        SET v_date_of_purchase = parse_date(p_dop);
    END IF;

    -- Get existing category or create new
    SELECT id INTO v_category_id 
    FROM categories 
    WHERE category_name = p_category 
    AND (subcategory_name = p_subcategory OR (subcategory_name IS NULL AND p_subcategory IS NULL));
    
    -- Get stone ID
    SELECT id INTO v_stone_id FROM stones WHERE stone_name = p_stone;
    
    -- Get color ID
    SELECT id INTO v_color_id FROM colors WHERE color_name = p_color;
    
    -- Get finish ID
    SELECT id INTO v_finish_id FROM finishes WHERE finish_name = p_finish;
      -- Insert inventory item
    INSERT INTO inventory_items (
        `SL`, `SKU Number`, category_id, `Weight`, length, width,
        stone_id, color_id, finish_id, `Year of Purchase`,
        `Status`, `CUSTOMER NAME`, `SALE AMOUNT`, `DOP`
    )
    VALUES (
        p_sl, p_sku, v_category_id, p_weight, p_length, p_width,
        v_stone_id, v_color_id, v_finish_id, p_year,
        p_status, p_customer, p_sale_amount, v_date_of_purchase
    );
    
    SET v_inventory_id = LAST_INSERT_ID();
    
    -- Insert pricing information
    INSERT INTO pricing (
        inventory_item_id, `SKU Number`, `Unit Price`, `Cost price`,
        `Thread work`, `GST on Cost price`, `Packaging cost`,
        `Final Cost price`, `SP - Margin`, `Taxes`, `SP`, `Final SP`
    )
    VALUES (
        v_inventory_id,
        p_sku,
        p_unit_price, 
        p_cost_price,
        p_thread_work, 
        p_gst, 
        p_packaging,
        p_final_cost,
        p_sp_margin,
        p_taxes,
        p_selling_price,
        p_final_sp
    );
END //

-- Create stored procedure to get inventory status
CREATE PROCEDURE get_inventory_status()
BEGIN
    SELECT 
        c.category_name,
        c.subcategory_name,
        COUNT(*) as total_items,
        SUM(CASE WHEN i.`Status` = 'In Stock' THEN 1 ELSE 0 END) as in_stock,
        SUM(CASE WHEN i.`Status` = 'OO Stock' THEN 1 ELSE 0 END) as out_of_stock,        SUM(p.`Final Cost price`) as total_inventory_cost,
        SUM(p.`Final SP`) as total_expected_value,
        SUM(CASE WHEN i.`Status` = 'OO Stock' THEN i.`SALE AMOUNT` ELSE p.`Final SP` END) as total_actual_value,
        AVG(CAST(REPLACE(p.`SP - Margin`, '%', '') AS DECIMAL(10,2))) as avg_margin,
        AVG(p.`Taxes`) as avg_taxes
    FROM 
        inventory_items i
        JOIN categories c ON i.category_id = c.id
        JOIN pricing p ON i.id = p.inventory_item_id
    GROUP BY 
        c.category_name, c.subcategory_name
    ORDER BY 
        c.category_name, c.subcategory_name;
END //

-- Create stored procedure to get sales report
CREATE PROCEDURE get_sales_report(
    IN start_date DATE,
    IN end_date DATE
)
BEGIN
    SELECT 
        i.`SKU Number`,
        c.category_name,
        c.subcategory_name,
        i.`CUSTOMER NAME`,
        i.`SALE AMOUNT`,
        i.`DOP`,        p.`Final Cost price`,
        p.`SP - Margin`,
        p.`Taxes`,
        p.`SP`,
        p.`Final SP`,
        (i.`SALE AMOUNT` - p.`Final Cost price`) as actual_profit,
        (p.`Final SP` - p.`Final Cost price`) as expected_profit
    FROM 
        inventory_items i
        JOIN categories c ON i.category_id = c.id
        JOIN pricing p ON i.id = p.inventory_item_id
    WHERE 
        i.`DOP` BETWEEN start_date AND end_date
        AND i.`Status` = 'OO Stock'
    ORDER BY 
        i.`DOP`;
END //

DELIMITER ;