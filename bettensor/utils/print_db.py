import sqlite3
def print_all_data(db_name='games_data.db'):
    # Connect to the database
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    # Execute a query to retrieve all data from the game_data table
    c.execute('''SELECT * FROM game_data''')
    # Fetch all rows from the executed query
    rows = c.fetchall()
    # Get the column names
    column_names = [description[0] for description in c.description]
    # Print the column names
    print("\t".join(column_names))
    # Print each row of data
    for row in rows:
        print("\t".join(str(item) for item in row))
    # Close the database connection
    conn.close()
# Call the function to print all data
print_all_data()