"""
Diagnostic script to check REPORTING schema access in SQL Server.
Uses environment variables for database credentials.
"""

import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables from .env file
load_dotenv()


def check_reporting_schema():
    """Check REPORTING schema and orders view/table existence and permissions."""

    # Get database credentials from environment
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USERNAME")
    db_password = os.getenv("DB_PASSWORD")
    db_driver = os.getenv("DB_DRIVER")
    print("HOST:", db_host)
    print("DB NAME:", db_name)
    print("USER:", db_user)
    print("PASSWORD:", db_password)
    print("DRIVER:", db_driver)
    if not all([db_host, db_name, db_user, db_password]):
        print("ERROR: Missing required environment variables!")
        print("Required: DB_HOST, DB_NAME, DB_USERNAME, DB_PASSWORD")
        print("Optional: DB_DRIVER, DATABASE_URI")
        sys.exit(1)

    # Build connection string with SQL Authentication
    conn_str = (
        f"mssql+pyodbc://{db_user}:{db_password}@"
        f"{db_host}/{db_name}?driver={db_driver}&TrustServerCertificate=yes"
    )
    print(conn_str)
    print(f"Connecting to: {db_host}/{db_name}")
    print(f"User: {db_user}")
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
                print("   ✓ REPORTING schema EXISTS")
            else:
                print("   ✗ REPORTING schema NOT FOUND")
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
                print("   ✗ REPORTING.orders NOT FOUND")
                print("   → Check if it exists in another schema")

            # 3. List all objects in dbo schema
            print("\n3. All objects in dbo schema:")
            result = conn.execute(
                text(
                    """
                SELECT TABLE_TYPE, TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'dbo'
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

            # 5. Try to query REPORTING.orders if it exists
            if orders_exists:
                print("\n5. Testing SELECT access to REPORTING.orders...")
                try:
                    result = conn.execute(text("SELECT TOP 1 * FROM REPORTING.orders"))
                    row = result.fetchone()
                    if row:
                        print("   ✓ Successfully queried REPORTING.orders")
                        print(f"   Columns: {result.keys()}")
                    else:
                        print("   ✓ Access granted (table/view is empty)")
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
