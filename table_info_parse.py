import json
from typing import TypedDict, NotRequired, cast


# MARK: Type Definitions
class ColumnInfo(TypedDict):
    name: str
    desc: str


class TableData(TypedDict):
    description: str
    usage_notes: NotRequired[str]
    columns: NotRequired[list[ColumnInfo]]
    business_rules: NotRequired[list[str]]
    common_queries: NotRequired[list[str]]


# MARK: Formatting Utilities
def format_table_info(table_data: TableData) -> str:
    """Convert JSON structure to formatted string."""
    lines = [f"Table: {table_data['description']}"]

    usage_notes = table_data.get("usage_notes")
    if usage_notes:
        lines.append(f"\nUsage: {usage_notes}")

    columns = table_data.get("columns")
    if columns:
        lines.append("\nImportant Columns:")
        for col in columns:
            lines.append(f"  - {col['name']}: {col['desc']}")

    business_rules = table_data.get("business_rules")
    if business_rules:
        lines.append("\nBusiness Rules:")
        for rule in business_rules:
            lines.append(f"  - {rule}")

    common_queries = table_data.get("common_queries")
    if common_queries:
        lines.append("\nCommon Query Patterns:")
        for query in common_queries:
            lines.append(f"  - {query}")

    return "\n".join(lines)


# MARK: Main Execution
# Load from file
with open("table_info.json", "r", encoding="utf-8") as f:
    schema_json = cast(dict[str, TableData], json.load(f))

# Convert to custom_table_info format
custom_table_info = {
    table_name: format_table_info(table_data)
    for table_name, table_data in schema_json.items()
}

# print with new lines
for table, info in custom_table_info.items():
    print(f'"{table}": """\n{info}\n""",')
