# Custom Schema Mode Implementation

## Overview

The agent now operates in **Custom Schema Mode**, which means it does **NOT** read actual table structures from the database. Instead, it relies entirely on the schema information you provide in `db_schema_config.json`.

## What Changed

### Before (Old Behavior)

- SQLDatabase would connect and read actual table metadata from the database
- Agent could see actual column names, data types, indexes, and sample rows
- `sample_rows_in_table_info=2` would fetch 2 sample rows from each table
- Agent had visibility into the real database structure

### After (New Behavior - Custom Schema Mode)

- SQLDatabase receives **only** the information from `db_schema_config.json`
- Agent sees **only** what you explicitly define in the JSON file
- No actual table metadata is read from the database
- No sample rows are fetched
- The database structure remains completely hidden from the agent

## Benefits

### 🔒 Enhanced Security

- Agent never sees the actual database structure
- Real column names, types, and constraints remain hidden
- You control exactly what information the agent receives

### 🛡️ Privacy Protection

- No actual data samples are exposed to the agent
- Database schema details are abstracted through your descriptions
- Perfect for sensitive databases where structure itself is confidential

### 📝 Documentation Control

- You define the "view" of the database the agent sees
- Can simplify complex schemas with plain English descriptions
- Can hide technical implementation details

### 🎯 Reduced Database Load

- No metadata queries executed against the database
- Faster initialization
- Lower database resource usage

## How It Works

### 1. Configuration in `db_schema_config.json`

```json
{
  "dbo.users": {
    "description": "Contains user account information and registration details",
    "columns": {
      "id": "Primary key, unique user identifier",
      "name": "Full name of the user",
      "email": "Email address for login and communication",
      "created_at": "Registration date and time"
    },
    "foreign_keys": []
  }
}
```

### 2. Schema Processing

The `DatabaseManager.generate_custom_table_info()` method converts this JSON into the format SQLDatabase expects:

```python
{
  "dbo.users": """Table: dbo.users
Description: Contains user account information and registration details

Columns:
  - id: Primary key, unique user identifier
  - name: Full name of the user
  - email: Email address for login and communication
  - created_at: Registration date and time"""
}
```

### 3. Database Initialization

```python
db = SQLDatabase.from_uri(
    db_uri,
    include_tables=matched_tables,
    sample_rows_in_table_info=0,  # No sample rows
    custom_table_info=custom_table_info,  # Use our custom schema
    engine_args=engine_args,
)
```

## What the Agent Sees

When the agent uses SQL tools, it receives information **exclusively** from your JSON configuration:

### `sql_db_list_tables` Tool

```
Allowed tables (restricted by configuration): dbo.users, dbo.orders, dbo.products
```

### `sql_db_schema` Tool

```
Table: dbo.users
Description: Contains user account information and registration details

Columns:
  - id: Primary key, unique user identifier
  - name: Full name of the user
  - email: Email address for login and communication
  - created_at: Registration date and time
```

### `sql_db_query` Tool

- Still executes actual SQL queries
- But the agent plans queries based only on the custom schema information
- Table access is validated before execution

## Important Notes

### ⚠️ Keep Your Schema Accurate

Since the agent relies entirely on `db_schema_config.json`, you must ensure:

- Column names match the actual database
- Descriptions are accurate and helpful
- Foreign key relationships are correctly documented
- All columns the agent needs to use are listed

### ⚠️ Schema Updates

When you modify your database schema, remember to update `db_schema_config.json`:

1. Add new tables to the JSON file
2. Update column definitions if changed
3. Document any new relationships
4. Remove tables/columns that should no longer be accessible

### ✅ Testing Queries

While the agent plans queries based on custom schema, the actual queries still execute against the real database. If column names in the JSON don't match the database, queries will fail.

## Logging

Look for these log messages to confirm custom schema mode is active:

```
INFO - Loaded 3 tables from custom schema: ['dbo.users', 'dbo.orders', 'dbo.products']
INFO - Using custom schema mode: Agent will NOT read actual table metadata from database
INFO - Using custom schema info for 3 tables
```

## Migration from Old Behavior

If you were previously relying on `sample_rows_in_table_info`:

**Before:**

```python
DatabaseManager(
    db_uri=db_uri,
    sample_rows=2,  # This fetched 2 sample rows
)
```

**After:**

```python
DatabaseManager(
    db_uri=db_uri,
    sample_rows=2,  # This parameter is now ignored (always 0 in custom mode)
)
```

The agent no longer sees sample rows. If you want to provide example data, add it to column descriptions:

```json
{
  "dbo.products": {
    "description": "Contains product catalog with inventory information",
    "columns": {
      "category": "Product category. Valid values: 'electronics', 'clothing', 'food'"
    }
  }
}
```

## Related Documentation

- [Table Access Control](./table_access_control.md) - How table access is enforced
- [Database Configuration](./database_connection_pooling.md) - Connection pool settings
- [Security & Safety](./security_safety.md) - Overall security practices

## Summary

✅ **Agent uses only `db_schema_config.json` for schema information**  
✅ **No actual database metadata is read**  
✅ **No sample rows are fetched**  
✅ **Enhanced security and privacy**  
✅ **You control exactly what the agent knows about your database**
