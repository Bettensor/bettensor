import requests

# Your API key
api_key = "b416b1c26dmsh6f20cd13ee1f7ccp11cc1djsnf64975aaacde"

# Define the endpoints and headers
fixtures_url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
odds_url = "https://api-football-v1.p.rapidapi.com/v3/odds"
headers = {
    "X-RapidAPI-Key": api_key,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
}

# Set the date for June 19, 2024
date = "2024-06-19"

# Define the query parameters for fixtures
fixtures_query = {
    "league": "253",  # MLS League ID
    "season": "2024",  # Current season
    "date": date,
}

# Make the API request to get fixtures
fixtures_response = requests.get(fixtures_url, headers=headers, params=fixtures_query)

# Check the response status
if fixtures_response.status_code == 200:
    fixtures = fixtures_response.json()
    # Iterate over each fixture to get the odds
    for fixture in fixtures["response"]:
        fixture_id = fixture["fixture"]["id"]
        print(
            f"Match: {fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}"
        )
        print(f"Date: {fixture['fixture']['date']}")
        print(f"Venue: {fixture['fixture']['venue']['name']}")

        # Define the query parameters for odds
        odds_query = {"fixture": fixture_id}

        # Make the API request to get odds
        odds_response = requests.get(odds_url, headers=headers, params=odds_query)

        # Check the response status
        if odds_response.status_code == 200:
            odds = odds_response.json()
            if odds["response"]:
                bookmaker_odds = odds["response"][0]["bookmakers"][0]["bets"][0][
                    "values"
                ]
                for odd in bookmaker_odds:
                    print(f"{odd['value']}: {odd['odd']}")
            else:
                print("No odds available for this match.")
        else:
            print(
                f"Error fetching odds: {odds_response.status_code} - {odds_response.text}"
            )

        print("-" * 30)
else:
    print(
        f"Error fetching fixtures: {fixtures_response.status_code} - {fixtures_response.text}"
    )
