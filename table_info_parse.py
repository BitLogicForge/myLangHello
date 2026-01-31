import json


def format_table_info(table_data):
    """Convert JSON structure to formatted string."""
    lines = [f"Table: {table_data['description']}"]

    if table_data.get("usage_notes"):
        lines.append(f"\nUsage: {table_data['usage_notes']}")

    if table_data.get("columns"):
        lines.append("\nImportant Columns:")
        for col in table_data["columns"]:
            lines.append(f"  - {col['name']}: {col['desc']}")

    if table_data.get("business_rules"):
        lines.append("\nBusiness Rules:")
        for rule in table_data["business_rules"]:
            lines.append(f"  - {rule}")

    if table_data.get("common_queries"):
        lines.append("\nCommon Query Patterns:")
        for query in table_data["common_queries"]:
            lines.append(f"  - {query}")

    return "\n".join(lines)


# Load from file
with open("table_info.json", "r", encoding="utf-8") as f:
    schema_json = json.load(f)

# Convert to custom_table_info format
custom_table_info = {
    table_name: format_table_info(table_data) for table_name, table_data in schema_json.items()
}

# print with new lines
for table, info in custom_table_info.items():
    print(f'"{table}": """\n{info}\n""",')
