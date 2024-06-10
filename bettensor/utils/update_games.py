import sqlite3
import json

# Convert odds to JSON string for storage
for game in data:
    game['odds'] = json.dumps(game['odds'])

# Connect to SQLite database
def update_games():
    conn = sqlite3.connect('games.db')
    cursor = conn.cursor()

    # Create game_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_data (
            homeTeam TEXT,
            awayTeam TEXT,
            game_id INTEGER PRIMARY KEY,
            date TEXT,
            odds TEXT
        )
    ''')

    # Insert data into game_data table if it does not already exist
    for game in data:
        cursor.execute('SELECT COUNT(1) FROM game_data WHERE game_id = ?', (game['game_id'],))
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
                INSERT INTO game_data (homeTeam, awayTeam, game_id, date, odds)
                VALUES (?, ?, ?, ?, ?)
            ''', (game['home'], game['away'], game['game_id'], game['date'], game['odds']))

    # Commit and close the connection
    conn.commit()
    conn.close()




    conn = sqlite3.connect('games.db')
    cursor = conn.cursor()

    # Fetch all records from game_data table
    cursor.execute('SELECT * FROM game_data')
    rows = cursor.fetchall()

    # Print all records
    for row in rows:
        homeTeam, awayTeam, game_id, date, odds = row
        # Convert odds from JSON string back to a dictionary
        odds = json.loads(odds)
        print(f'Home Team: {homeTeam}, Away Team: {awayTeam}, Game ID: {game_id}, Date: {date}, Odds: {odds}')

    # Close the connection
    conn.close()