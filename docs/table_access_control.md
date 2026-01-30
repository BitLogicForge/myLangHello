# Table Access Control Implementation

## Overview

The agent now enforces strict table access control based on the `include_tables` list defined in `db_schema_config.json`. The agent can only access tables that are explicitly listed in the configuration file and will be blocked from reading any other tables.

## Implementation Details

### 1. Database Manager Enhancements ([database_manager.py](../services/database_manager.py))

#### New Methods:

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
  "dbo.users": { ... },
  "dbo.orders": { ... },
  "dbo.products": { ... }
}
```

The keys of this JSON file determine which tables the agent can access. Any tables not listed here will be inaccessible to the agent.

## Security Benefits

1. **Prevents Information Disclosure**: Agent cannot discover or access tables outside the allowed list
2. **Defense in Depth**: Multiple layers of protection (query validation, schema restriction, list filtering)
3. **Clear Error Messages**: Users understand which tables are allowed when access is denied
4. **Audit Trail**: All access denials are logged for security monitoring

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
