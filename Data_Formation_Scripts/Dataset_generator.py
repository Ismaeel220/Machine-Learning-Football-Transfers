import pandas as pd
import numpy as np
from datetime import datetime

# define variables for seasons used for training and testing 
training_seasons = [2021, 2022] 
testing_seasons = [2023]
all_seasons = training_seasons + testing_seasons

target_leagues = ['GB1', 'IT1', 'ES1', 'L1'] 

# Global Football Rankings (Top 76)
league_coeffecients = {
    'GB1': 1.000,   'IT1': 0.934,   'ES1': 0.928,   'L1':  0.925,   'FR1': 0.924,
    'BE1': 0.883,   'GB2': 0.876,   'PO1': 0.872,   'BRA1': 0.869,  'MLS1': 0.861,
    'NL1': 0.852,   'DK1': 0.850,   'PL1': 0.850,   'AR1N': 0.849,  'JAP1': 0.847,
    'SE1': 0.843,   'KR1': 0.841,   'TR1': 0.838,   'MEXA': 0.836,  'ES2': 0.835,
    'NO1': 0.833,   'AT1': 0.831,   'RU1': 0.831,   'CYP1': 0.830,  'C1':  0.830,
    'EC1': 0.826,   'TS1': 0.826,   'COL1': 0.825,  'IT2': 0.824,   'L2':  0.823,
    'UNG1': 0.822,  'GR1': 0.822,   'SC1': 0.820,   'FR2': 0.819,   'KOR1': 0.818,
    'SA1': 0.816,   'MAR1': 0.810,  'RO1': 0.809,   'ALG1': 0.807,  'PAR1': 0.806,
    'GB3': 0.800,   'ISR1': 0.798,  'URU1': 0.797,  'EGY1': 0.793,  'CRC1': 0.791,
    'CHL1': 0.789,  'SLO1': 0.789,  'SVN1': 0.782,  'IRN1': 0.781,  'BRA2': 0.781,
    'BOL1': 0.779,  'UAE1': 0.777,  'RSA1': 0.776,  'AUS1': 0.775,  'AZ1': 0.775,
    'UKR1': 0.775,  'SER1': 0.770,  'PER1': 0.769,  'PO2': 0.768,   'BUL1': 0.767,
    'HON1': 0.760,  'BOS1': 0.759,  'TUN1': 0.750,  'BE2': 0.742,   'QAT1': 0.740,
    'FI1': 0.738,   'GB4': 0.737,   'GEO1': 0.735,  'CSL': 0.734,   'VEN1': 0.734,
    'NL2': 0.717,   'LET1': 0.704,  'SC2': 0.704,   'CAN1': 0.693,  'US2': 0.691,
    'IND1': 0.656
}
default_coefficient = 0.597

# Official FIFA Men's World Ranking (Top 13)
tier_1_nations= [
    'Spain', 'Argentina', 'France', 'England', 'Brazil', 
    'Portugal', 'Netherlands', 'Morocco', 'Belgium', 
    'Germany', 'Croatia', 'Senegal', 'Italy'
]

# load data 
print("Loading Raw Data...")
try:
    transfers = pd.read_csv('./Raw_data/transfers.csv')
    appearances = pd.read_csv('./Raw_data/appearances.csv')
    games = pd.read_csv('./Raw_data/games.csv')
    players = pd.read_csv('./Raw_data/players.csv')
    clubs = pd.read_csv('./Raw_data/clubs.csv')
    valuations = pd.read_csv('./Raw_data/player_valuations.csv')
except FileNotFoundError as e:
    print(f"Error: Missing file {e.filename}. Make sure all CSVs are in the same folder as this script.")
    exit()
if 'transfer_season' in transfers.columns:
    transfers['season'] = transfers['transfer_season']
elif 'Season' in transfers.columns:
    transfers['season'] = transfers['Season']
elif 'year' in transfers.columns:
    transfers['season'] = transfers['year']

# Converts the text "21/22" into the number 2021 so our math (season - 1) works!
transfers['season'] = transfers['season'].astype(str).str.split('/').str[0].astype(int) + 2000   

def get_season_stats(player_ids, season_year):
    # get matches from the specific season we are scouting 
    season_games = games[games['season'] == season_year]
    
    # If there are no games found, return empty so the whole script doesn't crash
    if season_games.empty: 
        return pd.DataFrame(columns=['player_id', 'adjusted_goals', 'adjusted_assists', 'adjusted_clean_sheet', 'goals_conceded', 'minutes_played'])
    
    # Link the player's personal match minutes to the actual final score
    apps = pd.merge(appearances[appearances['player_id'].isin(player_ids)], 
                    season_games[['game_id', 'home_club_goals', 'away_club_goals', 'home_club_id', 'away_club_id']], 
                    on='game_id')
    
    # A defender gets a clean sheet if they played >= 60 mins AND their team didn't concede    
    apps['is_cleansheet'] = np.where(
        (apps['minutes_played'] >= 60) & (
            ((apps['player_club_id'] == apps['home_club_id']) & (apps['away_club_goals'] == 0)) |
            ((apps['player_club_id'] == apps['away_club_id']) & (apps['home_club_goals'] == 0))
        ), 1, 0
    )
    
    # Calculate how many goals their team let in during that match
    apps['goals_conceded'] = np.where(
        apps['player_club_id'] == apps['home_club_id'], 
        apps['away_club_goals'],  
        apps['home_club_goals']   
    )
    
    # adjust stats based on league difficulty
    c = apps['competition_id'].map(league_coeffecients).fillna(default_coefficient)
    apps['adjusted_goals'] = apps['goals'] * c
    apps['adjusted_assists'] = apps['assists'] * c
    apps['adjusted_clean_sheet'] = apps['is_cleansheet'] * c
    
    # add all match by match stats to get totals for the whole season 
    return apps.groupby('player_id').agg({
        'adjusted_goals': 'sum', 
        'adjusted_assists': 'sum', 
        'adjusted_clean_sheet': 'sum', 
        'goals_conceded': 'sum',
        'minutes_played': 'sum'
    }).reset_index()





# stores finished data for each season
all_processed_seasons = []

# loop through each season one by one 
for season in all_seasons:
    print(f"loading {season} transfers...")

    # find clubs in the top target leagues
    target_club_ids = clubs[clubs['domestic_competition_id'].isin(target_leagues)]['club_id']
    
    # ignore free transfers and loans 
    season_transfers = transfers[(transfers['season'] == season) & 
                        (transfers['to_club_id'].isin(target_club_ids)) & 
                        (transfers['transfer_fee'] > 0)].copy()
    
    if season_transfers.empty: continue
    
    player_ids = season_transfers['player_id'].unique()
    
    # get the stats
    stats_player_season_1 = get_season_stats(player_ids, season - 1)
    stats_player_season_2 = get_season_stats(player_ids, season - 2)
    

    
    # Merge
    season_transfers = season_transfers.merge(stats_player_season_1, on='player_id', how='left')
    season_transfers = season_transfers.merge(stats_player_season_2, on='player_id', how='left', suffixes=('_season1', '_season2'))
    
    all_processed_seasons.append(season_transfers)


full_data = pd.concat(all_processed_seasons)

# Add static player info
full_data = pd.merge(full_data, players[['player_id', 'name', 'position', 'height_in_cm', 'country_of_citizenship', 'foot', 'contract_expiration_date']], on='player_id', how='left')

# Contracts
full_data['season_start_date'] = pd.to_datetime(full_data['season'].astype(str) + '-08-01')
full_data['contract_expiration_date'] = pd.to_datetime(full_data['contract_expiration_date'], errors='coerce')
full_data['days_left_on_contract'] = (full_data['contract_expiration_date'] - full_data['season_start_date']).dt.days

full_data['days_left_on_contract'] = full_data['days_left_on_contract'].fillna(365)
full_data['days_left_on_contract'] = full_data['days_left_on_contract'].apply(lambda x: x if x > 0 else 365)

# Age and Encodings
full_data['age'] = full_data['season'] - pd.to_datetime(players.set_index('player_id').loc[full_data['player_id']]['date_of_birth'].values).year
full_data['is_tier1_nation'] = full_data['country_of_citizenship'].apply(lambda x: 1 if x in tier_1_nations else 0)
full_data['is_left_footed'] = full_data['foot'].apply(lambda x: 1 if x == 'left' else 0)

# Fix missing heights 
avg_height = full_data['height_in_cm'].median()
full_data['height_in_cm'] = full_data['height_in_cm'].fillna(avg_height)

# Goals Conceded Per 90 Minutes
full_data['goals_conceded_per_90_season1'] = np.where(full_data['minutes_played_season1'] > 0, 
                                     (full_data['goals_conceded_season1'] / full_data['minutes_played_season1']) * 90, 0)
full_data['goals_conceded_per_90_season2'] = np.where(full_data['minutes_played_season2'] > 0, 
                                     (full_data['goals_conceded_season2'] / full_data['minutes_played_season2']) * 90, 0)

# Fill Missing Data
perf_cols = [
    'adjusted_goals_season1', 'adjusted_assists_season1', 'adjusted_clean_sheet_season1', 'minutes_played_season1', 'goals_conceded_season1', 'goals_conceded_per_90_season1',
    'adjusted_goals_season2', 'adjusted_assists_season2', 'adjusted_clean_sheet_season2', 'minutes_played_season2', 'goals_conceded_season2', 'goals_conceded_per_90_season2',
    'market_value_in_eur'
]
full_data[perf_cols] = full_data[perf_cols].fillna(0)

# Drop any rows where we still don't have a market value
full_data = full_data[full_data['market_value_in_eur'] > 0]

# Final selection
final_cols = [
    'name', 'transfer_fee', 'market_value_in_eur', 'days_left_on_contract', 
    'age', 'position', 'height_in_cm', 'is_left_footed', 'is_tier1_nation',
    'adjusted_goals_season1', 'adjusted_assists_season1', 'adjusted_clean_sheet_season1', 'goals_conceded_per_90_season1', 'minutes_played_season1',
    'adjusted_goals_season2', 'adjusted_assists_season2', 'adjusted_clean_sheet_season2', 'goals_conceded_per_90_season2', 'minutes_played_season2',
    'season', 'to_club_name'
]

# SAVE
full_data[full_data['season'].isin(training_seasons)][final_cols].to_csv('training_data.csv', index=False)
full_data[full_data['season'].isin(testing_seasons)][final_cols].to_csv('testing_data.csv', index=False)

print("\nSUCCESS! Dataset compiled.")
print(f"Training Data: {len(full_data[full_data['season'].isin(training_seasons)])} rows.")