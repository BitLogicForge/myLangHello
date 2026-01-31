custom_table_info = {
    "users": """
Table: users - User account information
Columns:
  - id (INT, PK): Unique user identifier
  - name (VARCHAR(100)): Full name
  - email (VARCHAR(255), UNIQUE): Email address
  - created_at (DATETIME): Account creation timestamp
Foreign Keys: None
Indexes: idx_email on email column
Business Rules:
  - Email must be unique and valid
  - Created_at defaults to current timestamp
""",
    "orders": """
Table: orders - Customer purchase orders
Columns:
  - id (INT, PK): Order ID
  - user_id (INT, FK): References users.id
  - product_id (INT, FK): References products.id
  - quantity (INT): Number of items ordered
  - total_price (DECIMAL(10,2)): Total order amount
  - status (VARCHAR(20)): pending|completed|cancelled
  - order_date (DATETIME): When order was placed
Foreign Keys:
  - user_id → users.id (CASCADE on delete)
  - product_id → products.id (RESTRICT on delete)
Indexes: idx_user_id, idx_status, idx_order_date
Business Rules:
  - Quantity must be positive
  - Status defaults to 'pending'
""",
}
