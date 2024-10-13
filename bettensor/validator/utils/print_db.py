import sqlite3

conn = sqlite3.connect("data/validator.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for table in tables:
    table_name = table[0]
    print(f"Table: {table_name}")
    cursor.execute(f"SELECT * FROM {table_name};")
    rows = cursor.fetchall()
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [column[1] for column in cursor.fetchall()]
    print(" | ".join(columns))
    for row in rows:
        print(row)
    print("\n")
# Close the connection
conn.close()
