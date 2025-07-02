"""
Simplified NFL Betting Prediction Data Pipeline
==============================================
Uses only nfl_data_py package - no web scraping required!
Complete data collection and preprocessing in one clean script.

Requirements: pip install pandas numpy nfl_data_py

Output: ML-ready dataset for logistic regression, random forest, and SVM models
"""

import pandas as pd
import numpy as np

print("Using nfl_data_py for all data\n")

# STEP 1: Download all data using nfl_data_py
print("STEP 1: Downloading NFL data...")

try:
    import nfl_data_py as nfl
    
    # Get game results (including 2024 for latest data)
    print("  Downloading game schedules and results...")
    games = nfl.import_schedules([2020, 2021, 2022, 2023, 2024])
    print(f"  Downloaded {len(games)} games")
    
    # Get team weekly stats (5 years of data)
    print("  Downloading team weekly statistics...")
    weekly_stats = nfl.import_weekly_data([2020, 2021, 2022, 2023, 2024])
    print(f"  Downloaded {len(weekly_stats)} team-week records")
    
    # Check what columns are available
    print(f"  Available columns: {list(weekly_stats.columns)[:10]}...")  # Show first 10 columns
    
except ImportError:
    print("ERROR: nfl_data_py not installed. Please run: pip install nfl_data_py")
    exit()
except Exception as e:
    print(f"ERROR downloading data: {e}")
    exit()

# STEP 2: Create game-level data with target variable
print("\nSTEP 2: Preparing game data...")

# Create target variable for supervised learning
# Reason: Need binary win/loss labels for classification algorithms
games = games.dropna(subset=['home_score', 'away_score', 'home_team', 'away_team'])
games['home_win'] = (games['home_score'] > games['away_score']).astype(int)

print(f"  Games after cleaning: {len(games)}")
print(f"  Home win rate: {games['home_win'].mean():.1%}")

# STEP 3: Process team weekly statistics for season averages
print("\nSTEP 3: Processing team statistics...")

# Calculate season-level team statistics from weekly data
# Reason: Need season averages for team strength comparisons
def create_season_team_stats(weekly_df):
    """Create season-level team statistics from weekly data"""
    # Find the correct team column name (it might be 'recent_team', 'team_name', etc.)
    team_col = None
    possible_team_cols = ['team', 'recent_team', 'team_name', 'posteam', 'Team']
    
    for col in possible_team_cols:
        if col in weekly_df.columns:
            team_col = col
            break
    
    if team_col is None:
        print(f"  ERROR: Could not find team column. Available columns: {list(weekly_df.columns)}")
        return pd.DataFrame()
    
    print(f"  Using team column: '{team_col}'")
    
    # Get numeric columns for aggregation
    numeric_cols = weekly_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Common NFL stat columns (use what's available)
    stat_mapping = {
        'passing_yards': ['passing_yards', 'pass_yds', 'Passing Yds'],
        'rushing_yards': ['rushing_yards', 'rush_yds', 'Rushing Yds'],
        'points_scored': ['points_scored', 'points', 'Points'],
        'total_yards': ['total_yards', 'tot_yds', 'Total Yds'],
        'points_allowed': ['points_allowed', 'opp_points', 'Opp Points'],
        'yards_allowed': ['yards_allowed', 'opp_tot_yds', 'Opp Total Yds']
    }
    
    # Find available columns
    agg_dict = {}
    available_stats = {}
    
    for stat_name, possible_cols in stat_mapping.items():
        for col in possible_cols:
            if col in weekly_df.columns:
                agg_dict[col] = 'mean'
                available_stats[stat_name] = col
                break
    
    if not agg_dict:
        print("  WARNING: No recognizable stat columns found. Using basic numeric aggregation.")
        # Use first few numeric columns
        for col in numeric_cols[:6]:
            if col != 'season' and col != 'week':
                agg_dict[col] = 'mean'
    
    print(f"  Aggregating stats: {list(agg_dict.keys())}")
    
    # Group by season and team
    season_stats = weekly_df.groupby(['season', team_col]).agg(agg_dict).reset_index()
    season_stats = season_stats.rename(columns={team_col: 'team'})
    
    # Create basic differentials if we have the right columns
    if 'points_scored' in available_stats and 'points_allowed' in available_stats:
        points_col = available_stats['points_scored']
        points_allowed_col = available_stats['points_allowed']
        season_stats['point_differential'] = season_stats[points_col] - season_stats[points_allowed_col]
    
    if 'total_yards' in available_stats and 'yards_allowed' in available_stats:
        yards_col = available_stats['total_yards']
        yards_allowed_col = available_stats['yards_allowed']
        season_stats['yards_differential'] = season_stats[yards_col] - season_stats[yards_allowed_col]
    
    return season_stats

team_stats = create_season_team_stats(weekly_stats)
print(f"  Team season stats created: {len(team_stats)} team-season records")

# STEP 4: Calculate 5-game rolling averages
print("\nSTEP 4: Calculating 5-game rolling averages...")

# Calculate rolling averages for recent performance trends
# Reason: Requires rolling averages for momentum analysis
def calculate_rolling_averages(games_df, window=5):
    """Calculate rolling averages for each team"""
    rolling_stats = []
    
    all_teams = list(set(games_df['home_team'].unique()) | set(games_df['away_team'].unique()))
    
    for team in all_teams:
        team_games = []
        
        # Get all games for this team (home and away)
        for _, game in games_df.sort_values(['season', 'week']).iterrows():
            if game['home_team'] == team:
                team_games.append({
                    'season': game['season'],
                    'week': game['week'],
                    'team': team,
                    'points_scored': game['home_score'],
                    'points_allowed': game['away_score'],
                    'won': game['home_win']
                })
            elif game['away_team'] == team:
                team_games.append({
                    'season': game['season'],
                    'week': game['week'],
                    'team': team,
                    'points_scored': game['away_score'],
                    'points_allowed': game['home_score'],
                    'won': 1 - game['home_win']
                })
        
        # Calculate rolling averages
        if team_games:
            team_df = pd.DataFrame(team_games).sort_values(['season', 'week'])
            
            # Manual rolling calculation
            for i in range(len(team_df)):
                start_idx = max(0, i - window + 1)
                end_idx = i + 1
                
                recent_games = team_df.iloc[start_idx:end_idx]
                team_df.loc[team_df.index[i], 'rolling_win_pct'] = recent_games['won'].mean()
                team_df.loc[team_df.index[i], 'rolling_points_scored'] = recent_games['points_scored'].mean()
                team_df.loc[team_df.index[i], 'rolling_points_allowed'] = recent_games['points_allowed'].mean()
                team_df.loc[team_df.index[i], 'rolling_point_diff'] = (recent_games['points_scored'] - recent_games['points_allowed']).mean()
            
            rolling_stats.append(team_df)
    
    return pd.concat(rolling_stats, ignore_index=True) if rolling_stats else pd.DataFrame()

rolling_stats = calculate_rolling_averages(games)
print(f"  Rolling averages calculated: {len(rolling_stats)} team-week records")

# STEP 5: Create matchup features for each game
print("\nSTEP 5: Creating game-level matchup features...")

# Transform from team stats to game-level predictions
# Reason: ML algorithms need head-to-head comparisons, not separate team stats
def create_matchup_features(games_df, team_stats_df, rolling_stats_df):
    """Create matchup features combining team stats and rolling averages"""
    matchup_data = []
    
    for _, game in games_df.iterrows():
        season = game['season']
        week = game['week']
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Get season stats for both teams
        home_stats = get_team_season_stats(team_stats_df, home_team, season)
        away_stats = get_team_season_stats(team_stats_df, away_team, season)
        
        # Get rolling stats up to this week
        home_rolling = get_rolling_stats_for_week(rolling_stats_df, home_team, season, week)
        away_rolling = get_rolling_stats_for_week(rolling_stats_df, away_team, season, week)
        
        if home_stats is not None and away_stats is not None:
            # Get available stat columns dynamically
            point_diff_col = 'point_differential'
            yards_col = None
            yards_allowed_col = None
            
            # Find available columns
            for col in home_stats.keys():
                if 'yards' in col.lower() and 'total' in col.lower():
                    yards_col = col
                elif 'yards' in col.lower() and ('allow' in col.lower() or 'opp' in col.lower()):
                    yards_allowed_col = col
            
            matchup_row = {
                'season': season,
                'week': week,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': game['home_score'],
                'away_score': game['away_score'],
                'home_win': game['home_win'],  # Target variable
                
                # Team strength differentials (use available columns)
                'home_point_diff': home_stats.get(point_diff_col, 0),
                'away_point_diff': away_stats.get(point_diff_col, 0),
                'point_diff_advantage': home_stats.get(point_diff_col, 0) - away_stats.get(point_diff_col, 0),
            }
            
            # Add yards features if available
            if yards_col and yards_allowed_col:
                matchup_row.update({
                    'home_off_vs_away_def': home_stats.get(yards_col, 0) - away_stats.get(yards_allowed_col, 0),
                    'away_off_vs_home_def': away_stats.get(yards_col, 0) - home_stats.get(yards_allowed_col, 0),
                    'yards_advantage': (home_stats.get(yards_col, 0) - away_stats.get(yards_allowed_col, 0)) - 
                                     (away_stats.get(yards_col, 0) - home_stats.get(yards_allowed_col, 0))
                })
            else:
                # Use basic stats if yards not available
                matchup_row.update({
                    'home_off_vs_away_def': 0,
                    'away_off_vs_home_def': 0,
                    'yards_advantage': 0
                })
            
            # Rolling performance features
            matchup_row.update({
                'home_rolling_win_pct': home_rolling.get('rolling_win_pct', 0.5) if home_rolling else 0.5,
                'away_rolling_win_pct': away_rolling.get('rolling_win_pct', 0.5) if away_rolling else 0.5,
                'home_rolling_points': home_rolling.get('rolling_points_scored', 20) if home_rolling else 20,
                'away_rolling_points': away_rolling.get('rolling_points_scored', 20) if away_rolling else 20,
                'rolling_win_advantage': (home_rolling.get('rolling_win_pct', 0.5) if home_rolling else 0.5) - 
                                       (away_rolling.get('rolling_win_pct', 0.5) if away_rolling else 0.5),
                'rolling_point_diff_advantage': (home_rolling.get('rolling_point_diff', 0) if home_rolling else 0) - 
                                               (away_rolling.get('rolling_point_diff', 0) if away_rolling else 0)
            })
            
            matchup_data.append(matchup_row)
    
    return pd.DataFrame(matchup_data)

def get_team_season_stats(team_stats_df, team_name, season):
    """Get team's season stats - handles dynamic column names"""
    team_data = team_stats_df[(team_stats_df['team'] == team_name) & (team_stats_df['season'] == season)]
    if not team_data.empty:
        return team_data.iloc[0].to_dict()
    return None

def get_rolling_stats_for_week(rolling_stats_df, team_name, season, week):
    """Get rolling stats before this week to prevent data leakage"""
    if rolling_stats_df.empty:
        return None
        
    team_data = rolling_stats_df[
        (rolling_stats_df['team'] == team_name) & 
        (rolling_stats_df['season'] == season) & 
        (rolling_stats_df['week'] < week)
    ]
    
    return team_data.iloc[-1].to_dict() if not team_data.empty else None

matchup_df = create_matchup_features(games, team_stats, rolling_stats)
print(f"  Matchup features created: {len(matchup_df)} game predictions")

if matchup_df.empty:
    print("ERROR: No matchup features created. Check data compatibility.")
    exit()

# STEP 6: Add team encoding and prepare for ML
print("\nSTEP 6: Final machine learning preparation...")

# One-hot encoding for team names
# Reason: ML algorithms need numerical team identifiers for home/away effects
home_encoded = pd.get_dummies(matchup_df['home_team'], prefix='home')
away_encoded = pd.get_dummies(matchup_df['away_team'], prefix='away')

# Combine all features
final_df = pd.concat([matchup_df, home_encoded, away_encoded], axis=1)
print(f"  Team encoding added: {len(home_encoded.columns)} home + {len(away_encoded.columns)} away features")

# Separate features from targets for supervised learning
# Reason: ML algorithms need X (features) and y (target) separation
feature_cols = [col for col in final_df.columns if col not in 
               ['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'home_win']]

X = final_df[feature_cols]
y = final_df['home_win']

# Code added: Standardize numerical features for ML algorithms
# Reason: Different scales (points vs percentages) need normalization for logistic regression and SVM
def manual_standardize(df, columns):
    """Standardize features using z-score normalization"""
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df[col] = (df[col] - mean_val) / std_val
    return df

numerical_cols = [
    'home_point_diff', 'away_point_diff', 'point_diff_advantage',
    'home_off_vs_away_def', 'away_off_vs_home_def', 'yards_advantage',
    'home_rolling_win_pct', 'away_rolling_win_pct',
    'home_rolling_points', 'away_rolling_points', 
    'rolling_win_advantage', 'rolling_point_diff_advantage'
]

X = manual_standardize(X, numerical_cols)

# STEP 7: Save all output files
print("\nSTEP 7: Saving final datasets...")

# Save ML-ready dataset files
# Reason: Create files ready for machine learning model training and evaluation
final_df.to_csv('complete_betting_dataset.csv', index=False)
X.to_csv('betting_features_X.csv', index=False)
y.to_csv('betting_targets_y.csv', index=False)

# Save feature names for reference
with open('feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_cols))

# Save intermediate files for documentation
team_stats.to_csv('team_season_stats.csv', index=False)
games.to_csv('clean_nfl_games.csv', index=False)

print("\n=== PIPELINE COMPLETE ===")
print(f"Final dataset shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")
print(f"Home win rate: {y.mean():.1%}")

print("\nFiles created:")
print("- complete_betting_dataset.csv (full dataset)")
print("- betting_features_X.csv (features for ML)")
print("- betting_targets_y.csv (win/loss targets)")
print("- feature_names.txt (feature documentation)")
print("- team_season_stats.csv (team statistics)")
print("- clean_nfl_games.csv (game results)")

print(f"\nFeature summary:")
print(f"- Numerical features: {len(numerical_cols)}")
print(f"- Team encoding features: {len(home_encoded.columns) + len(away_encoded.columns)}")
print(f"- Total features: {len(feature_cols)}")

print("\nAdvantages of this simplified approach:")
print("- No web scraping (faster, more reliable)")
print("- Single data source (consistent formatting)")
print("- Better data quality (official NFL data)")
print("- Easier maintenance (one package dependency)")

print("\nDataset ready for machine learning!")
print("Next steps:")
print("1. Load betting_features_X.csv and betting_targets_y.csv")
print("2. Split data chronologically for training/testing")
print("3. Train logistic regression, random forest, and SVM models")
print("4. Evaluate with accuracy, ROI simulation, and Brier score")