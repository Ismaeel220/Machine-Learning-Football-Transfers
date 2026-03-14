import pandas as pd
import numpy as np
import os
from datetime import datetime
import os
import shutil
from pathlib import Path

#CONFIGURATION AND SETUP OF DATA


#seasons for training data
training_seasons = [2020, 2021, 2022, 2023] 
#seasons for testing data
testing_seasons = [2024]
#all seasons from which data will be used
all_seasons = training_seasons + testing_seasons

#will be using data from the top 4 european leagues
target_leagues = ['GB1', 'IT1', 'ES1', 'L1'] 

# Global Football Rankings (Top 76)  used to weigh goals and assists
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
#defualt coefficient for leagues not in the above dictionary 
default_coefficient = 0.597

# Official FIFA Men's World Ranking (Top 13)
tier_1_nations= [
    'Spain', 'Argentina', 'France', 'England', 'Brazil', 
    'Portugal', 'Netherlands', 'Morocco', 'Belgium', 
    'Germany', 'Croatia', 'Senegal', 'Italy'
]


#LOAD DATA FROM RAW CSV FILES 


#try to load all the raw csv files if any file is missing to avoid crashing will simply return a message stating what file is missing
print("Loading Raw Data...")
try:
    #transfers is a csv file with every player who has moved to another team 
    transfers = pd.read_csv('./Raw_data/Transfrmarkt_data/transfers.csv')  
    #apperances is a csv file which contains infomation on every player and what games they played and for how long
    appearances = pd.read_csv('./Raw_data/Transfrmarkt_data/appearances.csv')
    #games contains scores for every match played, final score what teams scored and conceded etc
    games = pd.read_csv('./Raw_data/Transfrmarkt_data/games.csv')
    #information on every player, name date of birth, height etc
    players = pd.read_csv('./Raw_data/Transfrmarkt_data/players.csv')
    #linkseach c;ub to a league 
    clubs = pd.read_csv('./Raw_data/Transfrmarkt_data/clubs.csv')
    #tracks transfermarkts estimayes for how long much a player is worth over time 
    valuations = pd.read_csv('./Raw_data/Transfrmarkt_data/player_valuations.csv')
#if a file is not found then instead of crashing just print a message to the console explaining what file was not found
except FileNotFoundError as e:
    print(f"Error: Missing file {e.filename}. Make sure all CSVs are in the same folder as this script.")
    exit()


#in transfermarkt data the seasons are stored as 21/22 they are required to be stored as whole numbers for calculations later on therefore the value is
#  split into a list the first value obtained and then 2000 is added to go from 21/22 to e.g 2021 and save the result in a new column called season

transfers['season'] = transfers['transfer_season'].astype(str).str.split('/').str[0].astype(int) + 2000   



#method to get players stats from a specific year 
def get_season_stats(player_ids, season_year):
    #filter the games file to return only the games from the specific season we require and save them to the variable season_games
    season_games = games[games['season'] == season_year]
    
    #if the season has no games return an empty dataframe so rest of the script can run without crashing 
    if season_games.empty: 
        return pd.DataFrame(columns=['player_id', 'adjusted_goals', 'adjusted_assists', 'adjusted_clean_sheet', 'goals_conceded', 'minutes_played'])
    
    #merge apperance data actual game data for each player
    apps = pd.merge(appearances[appearances['player_id'].isin(player_ids)], 
                    season_games[['game_id', 'home_club_goals', 'away_club_goals', 'home_club_id', 'away_club_id']], 
                    on='game_id')
    
    
    #logic to make a cleansheet only count if a player has played more than 60 minuites in the game 
    #checks if the team was at home did the away team score 0 and vice versa and makesa column called is cleansheet
    apps['is_cleansheet'] = np.where(
        (apps['minutes_played'] >= 60) & (
            ((apps['player_club_id'] == apps['home_club_id']) & (apps['away_club_goals'] == 0)) |
            ((apps['player_club_id'] == apps['away_club_id']) & (apps['home_club_goals'] == 0))
        ), 1, 0
    )
    

#makes a column called goals conceeded which checks for how many goals the opposiion team scored based on who was home or away 
    apps['goals_conceded'] = np.where(
        apps['player_club_id'] == apps['home_club_id'], 
        apps['away_club_goals'],  
        apps['home_club_goals']   
    )
    

    #applies the league coefficient rankings to players goals assists and cleanshets
    c = apps['competition_id'].map(league_coeffecients).fillna(default_coefficient)
    apps['adjusted_goals'] = apps['goals'] * c
    apps['adjusted_assists'] = apps['assists'] * c
    apps['adjusted_clean_sheet'] = apps['is_cleansheet'] * c
    
    
    #calculates players statisitcs for a single season instead of having one row for every game that a player played in the season 
    return apps.groupby('player_id').agg({
        'adjusted_goals': 'sum', 
        'adjusted_assists': 'sum', 
        'adjusted_clean_sheet': 'sum', 
        'goals_conceded': 'sum',
        'minutes_played': 'sum'
    }).reset_index()





all_processed_seasons = []

#loop through the seasons for which we want data for 
for season in all_seasons:
    print(f"loading {season} transfers...")

    #find all the clubs in the top 4 leagues which we want 
    target_club_ids = clubs[clubs['domestic_competition_id'].isin(target_leagues)]['club_id']
    
    #transfer must be done in the current year , transfer fee greater than 0 so no free transfers or loans  and a player must be moving to a club in the top 4 leagues
    season_transfers = transfers[(transfers['season'] == season) & 
                        (transfers['to_club_id'].isin(target_club_ids)) & 
                        (transfers['transfer_fee'] > 0)].copy()
    
   
    if season_transfers.empty: continue
    
    #make a duplicate free list of all players who transferred
    player_ids = season_transfers['player_id'].unique()
    
    #gets the relevant stats for each player who transferref that year from the season before they transferred and two season before thet transferred
    stats_player_season_1 = get_season_stats(player_ids, season - 1)
    stats_player_season_2 = get_season_stats(player_ids, season - 2)
    

    #merge together season_transfers( details of the transfer that hapenned fee, player name etc)
    #with the stats of the player form the previous two seasons

    #merges season_transfers with stats from the season before 
    season_transfers = season_transfers.merge(stats_player_season_1, on='player_id', how='left')
    #merges the above with the stats from the season before since both contain the same column it wil apply appropriate suffixes to duplicate name dcolumns
    season_transfers = season_transfers.merge(stats_player_season_2, on='player_id', how='left', suffixes=('_season1', '_season2'))
    
    #add the processed season to the list 
    all_processed_seasons.append(season_transfers)

#once the loop is complete add all the seasons into one dataframe 
all_data = pd.concat(all_processed_seasons)


#right now the data only contains the info about transfer fees and match stats we now add the biographical info about the players
all_data = pd.merge(all_data, players[['player_id', 'name', 'position', 'height_in_cm', 'country_of_citizenship', 'foot', 'contract_expiration_date']], on='player_id', how='left')


#creates a constant season start date fro the calculation of how many days left on a players contract
#takes the year turns it to a string and adds the first of august to it and hen turns it into a date object so can be found on a calander
all_data['season_start_date'] = pd.to_datetime(all_data['season'].astype(str) + '-08-01')

#converts ghe expitation date on players contracts from the dataset to a date object  "coerce" if data is missing then it will be marked as a blank
all_data['contract_expiration_date'] = pd.to_datetime(all_data['contract_expiration_date'], errors='coerce')
# calculate how many days left on a players contract since we converted to date objects can caluclate the time between the dates and converts into a whole number of days
all_data['days_left_on_contract'] = (all_data['contract_expiration_date'] - all_data['season_start_date']).dt.days

#for missing data assume 365 days left on contract (1year)
all_data['days_left_on_contract'] = all_data['days_left_on_contract'].fillna(365)
#if the calculations resulted in any impossible numbers e.g negative or 0 then assume 365 days also 
all_data['days_left_on_contract'] = all_data['days_left_on_contract'].apply(lambda x: x if x > 0 else 365)


#calculate a player age by looking up the player by their id and then getting their date of birth 
# converting the text to a date and only keeping the year and then subtracting the birth year from the transfer year 
all_data['age'] = all_data['season'] - pd.to_datetime(players.set_index('player_id').loc[all_data['player_id']]['date_of_birth'].values).year

#adds a 1 if a player is from on eof the top footballing countries as defines in list at the start of script 
all_data['is_tier1_nation'] = all_data['country_of_citizenship'].apply(lambda x: 1 if x in tier_1_nations else 0)

#if player is left footed add a 1 if right footed add 0
all_data['is_left_footed'] = all_data['foot'].apply(lambda x: 1 if x == 'left' else 0)

# --- MOVED TO ML SCRIPT: fill in missing height values with the median ---
# To prevent Data Leakage, the ML script will calculate the median only from training data.


#normalises gaols conceeded by working out the number of goals conceeded per 90 minutes 
all_data['goals_conceded_per_90_season1'] = np.where(all_data['minutes_played_season1'] > 0, 
                                     (all_data['goals_conceded_season1'] / all_data['minutes_played_season1']) * 90, 0)
all_data['goals_conceded_per_90_season2'] = np.where(all_data['minutes_played_season2'] > 0, 
                                     (all_data['goals_conceded_season2'] / all_data['minutes_played_season2']) * 90, 0)

# takes the list of all the performance columns and replaces any missing values with 0 as if the data is missing presume that they didnt score or assist etc
perf_cols = [
    'adjusted_goals_season1', 'adjusted_assists_season1', 'adjusted_clean_sheet_season1', 'minutes_played_season1', 'goals_conceded_season1', 'goals_conceded_per_90_season1',
    'adjusted_goals_season2', 'adjusted_assists_season2', 'adjusted_clean_sheet_season2', 'minutes_played_season2', 'goals_conceded_season2', 'goals_conceded_per_90_season2',
]
all_data[perf_cols] = all_data[perf_cols].fillna(0)


#define list of final features that will be used for he dataset
final_cols = [
    'name', 'transfer_fee',  'days_left_on_contract', 
    'age', 'position', 'height_in_cm', 'is_left_footed', 'is_tier1_nation',
    'adjusted_goals_season1', 'adjusted_assists_season1', 'adjusted_clean_sheet_season1', 'goals_conceded_per_90_season1', 'minutes_played_season1',
    'adjusted_goals_season2', 'adjusted_assists_season2', 'adjusted_clean_sheet_season2', 'goals_conceded_per_90_season2', 'minutes_played_season2',
    'season', 'to_club_name'
]




all_data = all_data[final_cols]

print("\n--- Basic Datasets Compiled ---")
print(f"Total Base Players Extracted: {len(all_data)}")


#add advanced stats from fbref datasets

print("\nLoading FBref stats")



fbref_data_filepaths = [
    './Raw_Data/Fbref_data/cleaned_2019-20.csv', 
    './Raw_Data/Fbref_data/cleaned_2020-21.csv', 
    './Raw_Data/Fbref_data/cleaned_2021-22.csv', 
    './Raw_Data/Fbref_data/cleaned_2022-23.csv', 
    './Raw_Data/Fbref_data/cleaned_2023-24.csv'
]

fbref_data_collection = []

# Try to load the advanced stats information 
for path in fbref_data_filepaths:
    try:
        seasonal_df = pd.read_csv(path)
        fbref_data_collection.append(seasonal_df)
    except FileNotFoundError as e:
        # If the file is missing print exactly which one it is and shut down the script
        print(f"Error: Missing FBref file {e.filename}. Make sure all scouting CSVs are in the folder.")
        exit()


#combines all five seasons worth of advanced stats into one datatframe 
full_scouting_df = pd.concat(fbref_data_collection, ignore_index=True)

#rename the advanced stats player column to name so it matches with the transfermarkt data from earlier
full_scouting_df.rename(columns={'player': 'name'}, inplace=True)
#splits the season column in the same way done previously 
full_scouting_df['season'] = full_scouting_df['season'].astype(str).str.split('-').str[-1].astype(int)

#remove data from fbref that is not required since we already have it from the previous dataset
full_scouting_df = full_scouting_df.drop(columns=[c for c in ['rk', 'nation', 'pos', 'squad', 'comp', 'age', 'born'] if c in full_scouting_df.columns])
stats_to_keep = [col for col in full_scouting_df.columns if col not in ['name', 'season']]


#if players have dup;icate data remove 
full_scouting_df = full_scouting_df.drop_duplicates(subset=['name', 'season'])

print("Combining data...")

#combine transfer market and fbref data
transfer_dataset = pd.merge(all_data, full_scouting_df, on=['name', 'season'], how='left')
transfer_dataset['has_advanced_stats'] = transfer_dataset['Expected Goals'].notna().astype(int)


# destination folder path
output_dir = Path.cwd() /"training-testing-data"

# 2. Create the folder automatically if it doesn't exist
# parents=True creates any missing parent folders in the path
# exist_ok=True prevents an error if the folder already exists
output_dir.mkdir(parents=True, exist_ok=True)

# path where the file will live
file_path = output_dir /'transfer_dataset.csv'

# 4. Save the dataframe to that location
transfer_dataset.to_csv(file_path, index=False)

print(f"Raw full dataset generated and saved to: {file_path}")