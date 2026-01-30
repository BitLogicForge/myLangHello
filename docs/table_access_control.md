# Table Access Control Implementation

## Overview

The agent now enforces strict table access control based on the `include_tables` list defined in `db_schema_config.json`. The agent can **only** access tables that are explicitly listed in the configuration file and will be blocked from reading any other tables.

**Important:** The agent uses **custom schema mode**, which means it does NOT read actual table metadata from the database. Instead, it relies entirely on the schema information defined in `db_schema_config.json`. This provides:

- **Enhanced Security**: The agent never sees the actual database structure
- **Privacy Protection**: Actual table contents and structure remain hidden
- **Documentation Control**: You control exactly what information the agent receives about your database

## Implementation Details

### 1. Database Manager Enhancements ([database_manager.py](../services/database_manager.py))

#### New Methods:

- **`generate_custom_table_info() -> dict`**
  - Generates formatted table information from the custom schema JSON
  - Creates the `custom_table_info` dictionary used by SQLDatabase
  - Prevents SQLDatabase from reading actual table metadata
  - Returns formatted schema strings for each table

- **`is_table_allowed(table_name: str) -> bool`**
  - Checks if a table is in the allowed `include_tables` list
  - Performs case-insensitive matching
  - Supports both fully-qualified names (`dbo.users`) and simple names (`users`)
  - Returns `True` if table is allowed, `False` otherwise

- **`validate_table_access(table_name: str) -> None`**
  - Validates that a table is in the allowed list
  - Raises `ValueError` with descriptive message if access is denied
  - Lists all allowed tables in the error message

- **`_extract_table_names_from_query(query: str) -> list[str]`**
  - Extracts table names from SQL queries using regex patterns
  - Looks for tables after `FROM`, `JOIN`, `INTO`, and `UPDATE` keywords
  - Returns a list of unique table names found in the query
  - Used for validating query access before execution

#### Custom Schema Mode:

When initializing the database connection, the DatabaseManager now:

1. Loads table names and schema from `db_schema_config.json`
2. Generates formatted table information using `generate_custom_table_info()`
3. Passes this to SQLDatabase via the `custom_table_info` parameter
4. Sets `sample_rows_in_table_info=0` to prevent reading actual table data
5. The agent sees ONLY what's defined in the JSON file - no actual database metadata

### 2. Tools Manager Enhancements ([tools_manager.py](../services/tools_manager.py))

#### Updated Constructor:

- Added `db_manager` parameter to receive the DatabaseManager instance
- Passes the db_manager to SQL tool wrappers for validation

#### Enhanced SQL Tool Protection:

All SQL tools are now wrapped with table access control:

1. **`sql_db_query` (Query Tool)**
   - Validates all tables in the query before execution
   - Extracts table names from SQL query
   - Checks each table against the allowed list
   - Returns access denied error if unauthorized table is referenced
   - Still enforces read-only (SELECT only) policy
   - Still enforces output size limits

2. **`sql_db_schema` (Schema Tool)**
   - Validates requested table names before returning schema
   - Only shows schema for allowed tables
   - Returns access denied error for unauthorized tables

3. **`sql_db_list_tables` (List Tables Tool)**
   - Only returns the list of allowed tables from configuration
   - Hides all other database tables from the agent
   - Provides clear message that list is restricted

4. **`sql_db_query_checker`** (and other tools)
   - Pass through unchanged (no table access needed)

### 3. Agent Configurator Updates ([agent_configurator.py](../services/agent_configurator.py))

- Updated to pass `db_manager` instance to `ToolsManager`
- Enables table access validation throughout the agent lifecycle

## Configuration

Tables are configured in [db_schema_config.json](../db_schema_config.json):

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
  },
  "dbo.orders": { ... },
  "dbo.products": { ... }
}
```

**Key Points:**

- The **keys** of this JSON file determine which tables the agent can access
- Any tables not listed here will be completely invisible to the agent
- The agent receives ONLY the information defined in the JSON - it never reads actual table metadata
- This provides fine-grained control over what the agent knows about your database structure

## Security Benefits

1. **Complete Schema Isolation**: Agent never reads actual database metadata - it only knows what you tell it in the JSON config
2. **Prevents Information Disclosure**: Agent cannot discover or access tables outside the allowed list
3. **Defense in Depth**: Multiple layers of protection (custom schema mode, query validation, schema restriction, list filtering)
4. **Clear Error Messages**: Users understand which tables are allowed when access is denied
5. **Audit Trail**: All access denials are logged for security monitoring
6. **Privacy Protection**: Actual table structures, column types, and sample data remain completely hidden from the agent

## Example Scenarios

### Allowed Access

```
User: "Show me all users"
Agent: SELECT * FROM dbo.users
Result: ✅ Query executes successfully (users is in allowed list)
```

### Blocked Access

```
User: "Show me the admin_logs table"
Agent: SELECT * FROM admin_logs
Result: ❌ Access denied: Table 'admin_logs' is not in the allowed tables list.
         Allowed tables: [dbo.users, dbo.orders, dbo.products]
```

### Schema Protection

```
User: "What columns are in the sensitive_data table?"
Agent: sql_db_schema("sensitive_data")
Result: ❌ Access denied: Table 'sensitive_data' is not in the allowed tables list.
         Allowed tables: [dbo.users, dbo.orders, dbo.products]
```

### List Tables Protection

```
User: "What tables are available?"
Agent: sql_db_list_tables()
Result: ✅ Allowed tables (restricted by configuration): dbo.users, dbo.orders, dbo.products
```

## Testing

To test the table access control:

1. Query an allowed table:

   ```
   "How many users are in the database?"
   ```

2. Try to access a non-allowed table:

   ```
   "Show me data from the admin_logs table"
   ```

3. Request table list:

   ```
   "What tables can I query?"
   ```

4. Request schema for non-allowed table:
   ```
   "What is the schema of the system_config table?"
   ```

## Logging

All table access violations are logged at WARNING level:

- Query access denials: `"Table access denied: ..."`
- Schema access denials: `"Schema access denied for table: ..."`
- Blocked SQL commands: `"Blocked dangerous SQL command: ..."`

Check your application logs to monitor unauthorized access attempts.

## Limitations

The table name extraction from SQL queries uses regex patterns and may not catch:

- Complex subqueries with nested table references
- Dynamic SQL or stored procedure calls
- Table aliases that obscure the original table name

However, even if extraction misses some cases, the underlying SQLDatabase connection is already configured with `include_tables`, providing a secondary layer of protection at the database connection level.
