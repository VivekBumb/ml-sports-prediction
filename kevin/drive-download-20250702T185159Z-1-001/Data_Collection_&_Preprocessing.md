# Data Collection Process

## Overview

Our NFL betting prediction system uses the `nfl_data_py` package to collect official NFL data directly from league sources, eliminating the need for web scraping and ensuring reliable, consistent data quality.

## Data Sources

```python
# Primary data collection via nfl_data_py package
import nfl_data_py as nfl

# Two main data streams:
games = nfl.import_schedules([2020, 2021, 2022, 2023, 2024])       # Game results
weekly_stats = nfl.import_weekly_data([2020, 2021, 2022, 2023, 2024])  # Team stats
```

## Data Coverage

- **Time Period:** 2020-2024 NFL seasons (5 complete seasons)
- **Game Data:** 1,408 total games including playoffs
- **Team Statistics:** Weekly performance metrics for all 32 NFL teams
- **Data Quality:** Official NFL statistics (no web scraping required)

## Key Advantages

**Official Data:** Direct from NFL sources via nfl_data_py  
**No Web Scraping:** Reliable, maintained data pipeline  
**Consistent Format:** Standardized column names and data types  
**Real-time Updates:** Package automatically handles data formatting changes  
**Complete Coverage:** All games, teams, and seasons in one source  

## Data Download Script

```bash
# Run the data collection and preprocessing pipeline
python simplified_nfl_data.py
```
Note: Command could be python3 or depending upon the python version installed on the system.

## Dependencies

```bash
pip install nfl_data_py pandas numpy
```

## Output Files Generated

- `clean_nfl_games.csv` - Complete game results with scores and metadata

## Error Handling

The script includes robust error handling for:

- Missing `nfl_data_py` package installation
- Network connectivity issues during data download
- Data compatibility checks between seasons
- Column name variations across different data updates


## Data Freshness

- **Current dataset:** Through 2024 NFL season (complete)
- **For 2025 predictions:** Script can be easily updated to include new season data
- **Update frequency:** Run script after each NFL season concludes

<br>

# Data Preprocessing

## Overview

The preprocessing pipeline transforms raw NFL game data into ML ready features through statistical aggregation, feature engineering, and standardization. This follows the methodology outlined in our research proposal for optimal predictive performance.

## Preprocessing Steps

### 1. Data Cleaning & Validation

**Why we need it: Machine learning algorithms cannot handle missing values and require numerical targets for classification.**

```python
# Remove games with missing scores or team information
games = games.dropna(subset=['home_score', 'away_score', 'home_team', 'away_team'])

# Create binary target variable
games['home_win'] = (games['home_score'] > games['away_score']).astype(int)
```

### 2. Season-Level Team Statistics

**Why we need it: Season averages establish each team's baseline strength and overall quality, providing essential context for predicting individual game matchups.**

```python
# Aggregate weekly stats into season averages
season_stats = weekly_df.groupby(['season', 'team']).agg({
    'passing_yards': 'mean',
    'rushing_yards': 'mean', 
    'points_scored': 'mean',
    'points_allowed': 'mean'
}).reset_index()

# Calculate differentials
season_stats['point_differential'] = season_stats['points_scored'] - season_stats['points_allowed']
```

### 3. Rolling Performance Metrics 

**Why we need it: Recent team performance (momentum) is more predictive than season-long averages for sports outcomes.**

```python
# 5-game rolling averages for momentum analysis
def calculate_rolling_averages(games_df, window=5):
    for team in all_teams:
        # Calculate rolling statistics
        rolling_win_pct = recent_games['won'].rolling(window).mean()
        rolling_points_scored = recent_games['points_scored'].rolling(window).mean()
        rolling_point_diff = rolling_points_scored - rolling_points_allowed
```

**Rolling Features Created:**
- `home_rolling_win_pct` / `away_rolling_win_pct`
- `home_rolling_points` / `away_rolling_points`
- `rolling_win_advantage` / `rolling_point_diff_advantage`

### 4. Matchup-Level Feature Engineering

**Why we need it: Football outcomes depend on relative matchups, not absolute team strength - a good offense vs weak defense creates different dynamics than good offense vs strong defense.**

```python
# Transform team-level stats into head-to-head comparisons
matchup_features = {
    'point_diff_advantage': home_point_diff - away_point_diff,
    'home_off_vs_away_def': home_offense - away_defense,
    'yards_advantage': home_yards_advantage - away_yards_advantage
}
```

### 5. One-Hot Team Encoding 

**Why we need it: Machine learning algorithms require numerical inputs and cannot process categorical text data like team names (e.g., "Chiefs", "Bills").**

```python
# Create binary indicators for each team (home/away)
home_encoded = pd.get_dummies(matchup_df['home_team'], prefix='home')
away_encoded = pd.get_dummies(matchup_df['away_team'], prefix='away')

# Results in 64 team features (32 home + 32 away)
# Examples: home_KC, away_BUF, home_TB, etc.
```

### 6. Feature Standardization 

**Why we need it: Features with different scales (points vs percentages) would dominate the gradient descent optimization, preventing the algorithm from learning properly. Also, critical for logistic regression convergence**

```python
# Z-score normalization: (value - mean) / std
numerical_features = [
    'home_point_diff', 'away_point_diff', 'point_diff_advantage',
    'rolling_win_advantage', 'home_rolling_points', etc.
]

for feature in numerical_features:
    X[feature] = (X[feature] - X[feature].mean()) / X[feature].std()
```

## Feature Categories Created

### Performance Features (12 total)

- **Season Stats:** `home_point_diff`, `away_point_diff`, `point_diff_advantage`
- **Matchup Stats:** `home_off_vs_away_def`, `away_off_vs_home_def`, `yards_advantage`
- **Rolling Stats:** `home_rolling_win_pct`, `rolling_win_advantage`, `rolling_point_diff_advantage`

### Team Identity Features (64 total)

- **Home Teams:** `home_ARI`, `home_ATL`, `home_BAL`, ..., `home_WAS` (32 features)
- **Away Teams:** `away_ARI`, `away_ATL`, `away_BAL`, ..., `away_WAS` (32 features)

## Data Quality Assurance

### Missing Value Handling

```python
# Imputation strategy for early-season games
def get_rolling_stats_for_week(team, season, week):
    if week < 5:  # Insufficient rolling data
        return default_values  # Use league averages
    else:
        return actual_rolling_stats
```

### Data Leakage Prevention

```python
# Rolling stats use only PRIOR games to prevent future information
rolling_stats = team_data[team_data['week'] < current_week]
```

### Chronological Validation

```python
# Train/test split respects time ordering
split_idx = int(0.8 * len(X))  # 80% train, 20% test
X_train, X_test = X[:split_idx], X[split_idx:]  # No random shuffling
```

## Output Specifications

### Final Dataset Dimensions

- **Samples:** 1,408 NFL games (2020-2024)
- **Features:** 76 total features
  - 12 performance/statistical features
  - 64 team encoding features (32 home + 32 away)
- **Target:** Binary home team victory (0/1)

### Feature Scaling Results

- **Numerical features:** Mean ≈ 0, Standard deviation ≈ 1
- **Binary features:** Values in {0, 1}
- **No missing values:** Complete dataset ready for ML

## Output Files Generated

- `team_season_stats.csv` - Aggregated team performance by season
- `betting_features_X.csv` - ML-ready feature matrix (1,408 × 76)
- `betting_targets_y.csv` - Binary target variable (home team wins)
- `complete_betting_dataset.csv` - Full dataset with all features and targets
- `features_names.txt` - List of all 76 feature names

## Preprocessing Validation

### Data Integrity Checks

```python
No missing values in final dataset
All features properly scaled
Target distribution: 57.2% home wins (realistic)
Feature correlation analysis completed
Chronological ordering maintained
```

### Research Proposal Compliance

**5-game rolling averages:** Implemented as specified  
**One-hot team encoding:** All 32 teams encoded  
**Feature standardization:** Z-score normalization applied  
**Relative performance metrics:** Advantage calculations included  
**Data leakage prevention:** Temporal ordering respected  

## Preprocessing Runtime

- **Execution time:** ~30-60 seconds on standard hardware
- **Memory usage:** <500MB peak
- **Dependencies:** pandas, numpy only