import requests
import pandas as pd

# API details
API_KEY = 'b1a369409ee64b53ab507b443488c649'
BASE_URL = 'https://api.football-data.org/v4/'
headers = {'X-Auth-Token': API_KEY}

# Fetch competitions
response = requests.get(BASE_URL + 'competitions', headers=headers)
if response.status_code == 200:
    print("API connection successful!")
    competitions = response.json()
    print(competitions)
else:
    print(f"Error: {response.status_code} - {response.text}")

# Fetch matches for Premier League (PL)
competition_code = 'PL'
matches_url = BASE_URL + f'competitions/{competition_code}/matches'
response = requests.get(matches_url, headers=headers)

if response.status_code == 200:
    matches_data = response.json()
    print(matches_data)
else:
    print(f"Error: {response.status_code} - {response.text}")

# Process match data
matches = matches_data.get('matches', [])
match_records = []
for match in matches:
    match_records.append({
        'match_id': match['id'],
        'competition': match['competition']['name'],
        'home_team': match['homeTeam']['name'],
        'away_team': match['awayTeam']['name'],
        'home_score': match['score']['fullTime']['home'],
        'away_score': match['score']['fullTime']['away'],
        'match_date': match['utcDate'],
        'status': match['status']
    })

# Convert to DataFrame and save to CSV
df_matches = pd.DataFrame(match_records)
df_matches.to_csv('matches.csv', index=False)

# Fetch standings data
standings_url = BASE_URL + f'competitions/{competition_code}/standings'
response = requests.get(standings_url, headers=headers)

if response.status_code == 200:
    standings_data = response.json()
    print(standings_data)
else:
    print(f"Error: {response.status_code} - {response.text}")

# Process standings data
standings = standings_data.get('standings', [])[0]['table']
standings_records = []
for team in standings:
    standings_records.append({
        'team_name': team['team']['name'],
        'position': team['position'],
        'played_games': team['playedGames'],
        'won': team['won'],
        'draw': team['draw'],
        'lost': team['lost'],
        'points': team['points'],
        'goal_difference': team['goalDifference']
    })

# Save standings to CSV
df_standings = pd.DataFrame(standings_records)
df_standings.to_csv('standings.csv', index=False)

print("Data saved to 'matches.csv' and 'standings.csv'")
