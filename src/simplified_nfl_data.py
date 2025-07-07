import pandas as pd
import numpy as np

# get nfl data from nfl_data_py package
import nfl_data_py as nfl

# 5 years of data - should be enough
games = nfl.import_schedules([2020, 2021, 2022, 2023, 2024])
weekly = nfl.import_weekly_data([2020, 2021, 2022, 2023, 2024])

# clean the file, remove games with incomplete data
games = games.dropna(subset=['home_score', 'away_score'])
games['home_win'] = (games['home_score'] > games['away_score']).astype(int) # if Home team wins = 1 else if Away team wins (or ties) = 0

# print(games.columns)
print(f"got {len(games)} games")

# calculate each team strength by passing yards and rushing yards
def make_season_stats(data):
 
    # hanlde improrer naming of the column in the data
    team_col = 'recent_team' if 'recent_team' in data.columns else 'team'
    
    # season avarage for passing and rushing yards for each team
    result = data.groupby(['season', team_col]).agg({
        'passing_yards': 'mean',
        'rushing_yards': 'mean'
    }).reset_index()

    # standardize column name
    result.rename(columns={team_col: 'team'}, inplace=True)
    return result

# calculate the same for weekly data
season_data = make_season_stats(weekly)

# to calculate rolling avaerages with recent team performance data
rolling_stats = []

# get all team names
all_teams = list(set(games['home_team']) | set(games['away_team']))

for team in all_teams:
    team_games = []
    
    # get every game for this team 
    for idx, game in games.sort_values(['season', 'week']).iterrows():
        if game['home_team'] == team:
            team_games.append({
                'season': game['season'], 'week': game['week'], 'team': team,
                'pts_scored': game['home_score'], 'pts_allowed': game['away_score'],
                'win': game['home_win']
            })
        elif game['away_team'] == team:
            team_games.append({
                'season': game['season'], 'week': game['week'], 'team': team,
                'pts_scored': game['away_score'], 'pts_allowed': game['home_score'], 
                'win': 1 - game['home_win']  # flip it for away games
            })
    
    if len(team_games) == 0:
        print(f"no games for {team}?")  # debugging
        continue

    # create a table in chronological order   
    df = pd.DataFrame(team_games)
    
    # calculate 5 game rolling averages as per proposal
    for i in range(len(df)):
        if i < 4:
            window = df.iloc[:i+1]
        else:
            window = df.iloc[i-4:i+1]
            
        df.loc[df.index[i], 'roll_win_pct'] = window['win'].mean()
        df.loc[df.index[i], 'roll_pts_scored'] = window['pts_scored'].mean()
        df.loc[df.index[i], 'roll_pts_allowed'] = window['pts_allowed'].mean()
        df.loc[df.index[i], 'roll_pt_diff'] = (window['pts_scored'] - window['pts_allowed']).mean()
    
    rolling_stats.append(df)

roll_data = pd.concat(rolling_stats, ignore_index=True)

# sanity check
print(f"rolling data has {len(roll_data)} records")

# now build the actual features for each game
features = []
skipped = 0

for _, game in games.iterrows():
    yr, wk, home, away = game['season'], game['week'], game['home_team'], game['away_team']
    
    # season stats
    home_season = season_data[(season_data['team'] == home) & (season_data['season'] == yr)]
    away_season = season_data[(season_data['team'] == away) & (season_data['season'] == yr)]
    
    if home_season.empty or away_season.empty:
        skipped += 1
        continue  # skip if no data
    
    # rolling stats - only use games before current week to avoid lookahead bias
    home_roll = roll_data[(roll_data['team'] == home) & (roll_data['season'] == yr) & (roll_data['week'] < wk)]
    away_roll = roll_data[(roll_data['team'] == away) & (roll_data['season'] == yr) & (roll_data['week'] < wk)]
    
    # basic game info
    row = {
        'season': yr, 'week': wk, 'home_team': home, 'away_team': away,
        'home_score': game['home_score'], 'away_score': game['away_score'], 
        'home_win': game['home_win']
    }
    
    # season-based features (simplified - just using basic stats)
    h_pass = home_season['passing_yards'].iloc[0]
    h_rush = home_season['rushing_yards'].iloc[0]
    a_pass = away_season['passing_yards'].iloc[0] 
    a_rush = away_season['rushing_yards'].iloc[0]
    
    # these should be calculated from actual point differentials
    row['home_point_diff'] = 0  # placeholder - would need more data
    row['away_point_diff'] = 0
    row['point_diff_advantage'] = 0
    
    # very rough offensive vs defensive matchup
    row['home_off_vs_away_def'] = (h_pass + h_rush) - (a_pass + a_rush)  # rough approximation
    row['away_off_vs_home_def'] = (a_pass + a_rush) - (h_pass + h_rush)
    row['yards_advantage'] = (h_pass + h_rush) - (a_pass + a_rush)
    
    # rolling features - important
    if not home_roll.empty:
        row['home_rolling_win_pct'] = home_roll.iloc[-1]['roll_win_pct']
        row['home_rolling_points'] = home_roll.iloc[-1]['roll_pts_scored']
        h_roll_diff = home_roll.iloc[-1]['roll_pt_diff']
    else:
        #  early 2020 games won't have enough team data
        row['home_rolling_win_pct'] = 0.5  # assume average
        row['home_rolling_points'] = 22  # rough nfl average
        h_roll_diff = 0
        
    if not away_roll.empty:
        row['away_rolling_win_pct'] = away_roll.iloc[-1]['roll_win_pct'] 
        row['away_rolling_points'] = away_roll.iloc[-1]['roll_pts_scored']
        a_roll_diff = away_roll.iloc[-1]['roll_pt_diff']
    else:
        row['away_rolling_win_pct'] = 0.5
        row['away_rolling_points'] = 22
        a_roll_diff = 0
    
    # caculate which team has better momentum
    row['rolling_win_advantage'] = row['home_rolling_win_pct'] - row['away_rolling_win_pct']
    row['rolling_point_diff_advantage'] = h_roll_diff - a_roll_diff
    
    features.append(row)

print(f"skipped {skipped} games due to missing data")

# convert list of games to table 
df = pd.DataFrame(features)
print(f"built features for {len(df)} games")

# one hot encode teams, covert team names to binary
home_dummies = pd.get_dummies(df['home_team'], prefix='home')
away_dummies = pd.get_dummies(df['away_team'], prefix='away')

# Merging all data, original features + home team columns + away team columns
final_df = pd.concat([df, home_dummies, away_dummies], axis=1)

# split data into feature (X) and target variable y for training  
cols_to_drop = ['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'home_win']
X = final_df.drop(columns=cols_to_drop)
y = final_df['home_win']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# columns that need to be standarized - important for logistic regression
numeric_cols = ['home_point_diff', 'away_point_diff', 'point_diff_advantage',
                'home_off_vs_away_def', 'away_off_vs_home_def', 'yards_advantage',
                'home_rolling_win_pct', 'away_rolling_win_pct', 'home_rolling_points', 
                'away_rolling_points', 'rolling_win_advantage', 'rolling_point_diff_advantage']

for col in numeric_cols:
    if col in X.columns:
        mean_val = X[col].mean()
        std_val = X[col].std()
        if std_val > 0:  # avoid division by zero
            X[col] = (X[col] - mean_val) / std_val

# save everything
X.to_csv('betting_features_X.csv', index=False)
y.to_csv('betting_targets_y.csv', index=False)
final_df.to_csv('complete_betting_dataset.csv', index=False)

# save other files that the logistic regression script expects
with open('feature_names.txt', 'w') as f:
    for col in X.columns:
        f.write(col + '\n')

season_data.to_csv('team_season_stats.csv', index=False)
games.to_csv('clean_nfl_games.csv', index=False)

print(f"done. {X.shape[0]} games, {X.shape[1]} features")
print(f"home teams win {y.mean():.1%} of time")
print(X.columns.tolist()) # check what features we have in final dataset