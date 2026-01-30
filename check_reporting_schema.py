"""
Diagnostic script to check REPORTING schema access in SQL Server.
Uses environment variables for database credentials.
"""

import os
import sys
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text


def check_reporting_schema():
    """Check REPORTING schema and orders view/table existence and permissions."""

    # Get database credentials from environment
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")

    # Check if DATABASE_URI is set
    db_uri = os.getenv("DATABASE_URI")

    if not db_uri:
        if not all([db_host, db_name]):
            print("ERROR: Missing required environment variables!")
            print("Required: DB_HOST, DB_NAME")
            print("Optional: DB_USER, DB_PASSWORD, DB_DRIVER")
            sys.exit(1)

        # Build connection string
        if db_user and db_password:
            # SQL Authentication
            conn_str = (
                f"mssql+pyodbc://{quote_plus(db_user)}:{quote_plus(db_password)}@"
                f"{db_host}/{db_name}?driver={quote_plus(db_driver)}"
            )
        else:
            # Windows Authentication
            conn_str = (
                f"mssql+pyodbc://{db_host}/{db_name}"
                f"?driver={quote_plus(db_driver)}&trusted_connection=yes"
            )
    else:
        conn_str = db_uri

    print(f"Connecting to: {db_host}/{db_name}")
    print(f"User: {db_user or 'Windows Auth'}")
    print("-" * 60)

    try:
        engine = create_engine(conn_str)
        with engine.connect() as conn:

            # 1. Check if REPORTING schema exists
            print("\n1. Checking if REPORTING schema exists...")
            result = conn.execute(
                text(
                    """
                SELECT SCHEMA_NAME 
                FROM INFORMATION_SCHEMA.SCHEMATA 
                WHERE SCHEMA_NAME = 'REPORTING'
            """
                )
            )
            schema_exists = result.fetchone()
            if schema_exists:
                print(f"   ✓ REPORTING schema EXISTS")
            else:
                print(f"   ✗ REPORTING schema NOT FOUND")
                print("   → Create it with: CREATE SCHEMA REPORTING")

            # 2. Check if orders exists in REPORTING schema
            print("\n2. Checking if REPORTING.orders exists...")
            result = conn.execute(
                text(
                    """
                SELECT TABLE_TYPE, TABLE_SCHEMA, TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = 'REPORTING' AND TABLE_NAME = 'orders'
            """
                )
            )
            orders_exists = result.fetchone()
            if orders_exists:
                print(f"   ✓ Found: {orders_exists[0]} - {orders_exists[1]}.{orders_exists[2]}")
            else:
                print(f"   ✗ REPORTING.orders NOT FOUND")
                print("   → Check if it exists in another schema")

            # 3. List all objects in REPORTING schema
            print("\n3. All objects in REPORTING schema:")
            result = conn.execute(
                text(
                    """
                SELECT TABLE_TYPE, TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = 'REPORTING'
                ORDER BY TABLE_TYPE, TABLE_NAME
            """
                )
            )
            objects = result.fetchall()
            if objects:
                for obj in objects:
                    print(f"   - {obj[0]}: {obj[1]}")
            else:
                print("   (no objects found)")

            # 4. Check current user
            print("\n4. Current database user:")
            result = conn.execute(text("SELECT SYSTEM_USER, USER_NAME()"))
            user_info = result.fetchone()
            print(f"   System User: {user_info[0]}")
            print(f"   Database User: {user_info[1]}")

            # 5. Try to query REPORTING.orders if it exists
            if orders_exists:
                print("\n5. Testing SELECT access to REPORTING.orders...")
                try:
                    result = conn.execute(text("SELECT TOP 1 * FROM REPORTING.orders"))
                    row = result.fetchone()
                    if row:
                        print(f"   ✓ Successfully queried REPORTING.orders")
                        print(f"   Columns: {result.keys()}")
                    else:
                        print(f"   ✓ Access granted (table/view is empty)")
                except Exception as e:
                    print(f"   ✗ Query failed: {e}")
                    print("   → Check SELECT permissions on REPORTING schema")

            # 6. Check if orders exists in other schemas
            print("\n6. Searching for 'orders' in all schemas...")
            result = conn.execute(
                text(
                    """
                SELECT TABLE_TYPE, TABLE_SCHEMA, TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_NAME LIKE '%order%'
                ORDER BY TABLE_SCHEMA, TABLE_NAME
            """
                )
            )
            all_orders = result.fetchall()
            if all_orders:
                for obj in all_orders:
                    print(f"   - {obj[0]}: {obj[1]}.{obj[2]}")
            else:
                print("   (no objects matching 'order' found)")

        print("\n" + "=" * 60)
        print("Diagnostic complete!")

    except Exception as e:
        print(f"\n✗ CONNECTION FAILED: {e}")
        print("\nTroubleshooting:")
        print("1. Verify DB_HOST, DB_NAME are correct")
        print("2. Check DB_USER and DB_PASSWORD if using SQL auth")
        print("3. Ensure SQL Server is accessible from this machine")
        print("4. Check firewall rules and SQL Server configuration")
        sys.exit(1)


if __name__ == "__main__":
    check_reporting_schema()
