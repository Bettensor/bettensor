import requests
import json
import time

def get_baseball_data():
    url = "https://api-baseball.p.rapidapi.com/games"

    # Will change these to be inputs to the function, just testing for now
    querystring = {"league": "1", "season": "2024", "date": "2024-05-29"}

    # Yeah I put my API key in our github repo, what about it?
    headers = {
        "X-RapidAPI-Key": "b416b1c26dmsh6f20cd13ee1f7ccp11cc1djsnf64975aaacde",
        "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    response.raise_for_status()  # check for https errors
    games = response.json()

    games_list = [{
        "home": i['teams']['home']['name'],
        "away": i['teams']['away']['name'],
        "game_id": i['id'],
        "date": i['date'],
        "odds": get_game_odds(i['id'])  # calculates average between sportsbooks
    } for i in games['response']]
    
    # save data to a json file, may not do this in the future
    with open('games_data.json', 'w') as f:
        json.dump(games_list, f, indent=4)
    return games_list

def get_game_odds(game_id):
    url = "https://api-baseball.p.rapidapi.com/odds"

    querystring = {"game": game_id}

    headers = {
        "X-RapidAPI-Key": "b416b1c26dmsh6f20cd13ee1f7ccp11cc1djsnf64975aaacde",
        "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    response.raise_for_status()  

    odds_data = response.json()
    total_home_odds = 0
    total_away_odds = 0
    count = 0
    
    if odds_data['results'] > 0:
        bookmakers = odds_data['response'][0]['bookmakers']
        for bookmaker in bookmakers:
            for bet in bookmaker['bets']:
                if bet['name'] == "Home/Away":
                    if len(bet['values']) >= 2:
                        home_odds = float(bet['values'][0]['odd'])
                        away_odds = float(bet['values'][1]['odd'])
                        total_home_odds += home_odds
                        total_away_odds += away_odds
                        count += 1
                    elif len(bet['values']) == 1:
                        home_odds = float(bet['values'][0]['odd']) if bet['values'][0]['value'] == 'Home' else None
                        away_odds = float(bet['values'][0]['odd']) if bet['values'][0]['value'] == 'Away' else None
                        if home_odds is not None:
                            total_home_odds += home_odds
                            count += 1
                        if away_odds is not None:
                            total_away_odds += away_odds
                            count += 1
        # Sometimes there isnt data for a team, so I calculate it here. Some other times the data is absurd
        # and this seems to happen more often when some data is missing. Maybe we'll excluse missing values
        # also, betting lines shift pretty frequently. How often do we want to update the lines?
        if home_odds is None:
            home_odds = calculate_missing_odds(away_odds)
            print('ran')
            print(home_odds)
        if away_odds is None:
            away_odds = calculate_missing_odds(home_odds)

        if count > 0:
            avg_home_odds = total_home_odds / count
            avg_away_odds = total_away_odds / count
            return {
                "average_home_odds": round(avg_home_odds, 2), 
                "average_away_odds": round(avg_away_odds, 2)
            }
    
    return {"average_home_odds": None, "average_away_odds": None}

def calculate_missing_odds(known_odds):
    # This does not work for games where there can be a Tie
    # Calculate the probability of the away team winning
    probability_known_winning = 1 / known_odds
    
    probability_missing_winning = 1 - probability_known_winning
    missing_odds = 1 / probability_missing_winning
    
    return missing_odds

if __name__ == "__main__":
    get_baseball_data()
