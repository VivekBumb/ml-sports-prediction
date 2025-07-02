import seaborn as sns
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import plotly.express as px

def plot_feature_correlations(df, target_col='home_win'):


    # Drop only one-hot team columns like 'home_KC', 'away_NE'
    team_cols = [
        col for col in df.columns
        if re.fullmatch(r'(home|away)_[A-Z]{2,3}', col) and col != target_col
    ]

    # Drop team indicator columns
    num_df = df.drop(columns=team_cols, errors='ignore')

    # Keep only numeric features
    num_df = num_df.select_dtypes(include='number')

    # Drop rows with missing values in the target
    num_df = num_df.dropna(subset=[target_col])

    # Compute correlation with the target
    corr = num_df.corr(numeric_only=True)[target_col].sort_values(key=abs, ascending=False)

    # Plot top 10 correlated features (excluding the target itself)
    top_corr = corr[1:11]  # skip self-correlation
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_corr.values, y=top_corr.index, palette='coolwarm')
    plt.title(f'Top 10 Features Correlated with {target_col}')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.show(block=False)

def plot_team_trend(df, team_abbr='KC', stat='home_score'):
    #idea, not just graph the trend for one team, but for the entire league, or maybe compare it to average
    # Filter rows where team is playing at home
    team_home = df[df['home_team'] == team_abbr].copy()
    team_home['team_side'] = 'home'
    team_home['team_score'] = team_home[stat]

    # Filter rows where team is away
    team_away = df[df['away_team'] == team_abbr].copy()
    team_away['team_side'] = 'away'
    team_away['team_score'] = team_away[stat.replace('home_', 'away_')]

    # Combine home and away games
    team_df = pd.concat([team_home, team_away])
    team_df.sort_values(by=['season', 'week'], inplace=True)

    # Compute rolling average
    team_df['rolling_avg'] = team_df['team_score'].rolling(5).mean()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(team_df['rolling_avg'], label=f'{team_abbr} 5-Game Rolling Avg')
    plt.title(f'{team_abbr}: 5-Game Rolling Average of {stat}')
    plt.xlabel('Game Index')
    plt.ylabel(stat)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

def compare_teams(df, team1='KC', team2='PHI', stat='home_score'):
    #compare all metrics hopefully

    # Average performance for each team (home + away)
    t1_home = df[df['home_team'] == team1][stat].mean()
    t1_away = df[df['away_team'] == team1][stat.replace('home_', 'away_')].mean()
    t2_home = df[df['home_team'] == team2][stat].mean()
    t2_away = df[df['away_team'] == team2][stat.replace('home_', 'away_')].mean()

    t1_avg = np.mean([t1_home, t1_away])
    t2_avg = np.mean([t2_home, t2_away])

    # Bar plot
    plt.figure(figsize=(6, 4))
    plt.bar([team1, team2], [t1_avg, t2_avg], color=['blue', 'green'])
    plt.title(f'Average {stat.replace("home_", "")} Comparison')
    plt.ylabel(stat)
    plt.tight_layout()
    plt.show(block=False)
def plot_logistic_feature_importance(df, features, target='home_win'):
    #add more features?

    X = df[features].dropna()
    y = df.loc[X.index, target]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    # Coefficient bar plot
    coefs = pd.Series(model.coef_[0], index=features)
    top = coefs.abs().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=top.values, y=top.index, palette='coolwarm')
    plt.title("Top Logistic Regression Feature Importances")
    plt.xlabel("Absolute Coefficient Value")
    plt.tight_layout()
    plt.show(block=False)
def simulate_roi(results_df, threshold=0.55):


    # results_df should include: ['model_prob', 'home_win', 'game_id']
    df = results_df.copy()
    df['bet'] = df['model_prob'] > threshold
    df['bet_result'] = df['bet'] & (df['home_win'] == 1)  # won bet
    df['payout'] = df['bet_result'] * 1.909 - df['bet']  # -1 for losses, +0.909 for wins

    df['cumulative_roi'] = df['payout'].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(df['cumulative_roi'])
    plt.axhline(0, linestyle='--', color='red')
    plt.title(f'Cumulative ROI (Threshold = {threshold})')
    plt.xlabel('Game Index')
    plt.ylabel('Cumulative ROI ($)')
    plt.tight_layout()
    plt.show(block=False)








def create_team_stat_trend(df, team_abbr='KC', stat='home_score', window=5):
    # Filter games where the team played
    home_df = df[df['home_team'] == team_abbr].copy()
    away_df = df[df['away_team'] == team_abbr].copy()

    # Add a unified stat column
    home_df['team_side'] = 'home'
    home_df['stat'] = home_df[stat]

    # Adjust stat column name for away games (e.g., 'home_score' -> 'away_score')
    away_stat = stat.replace('home_', 'away_')
    away_df['team_side'] = 'away'
    away_df['stat'] = away_df[away_stat]

    # Combine and sort by season/week
    team_df = pd.concat([home_df, away_df], ignore_index=True)
    team_df = team_df.sort_values(by=['season', 'week'])

    # Rolling average
    team_df['rolling_stat'] = team_df['stat'].rolling(window=window).mean()

    # Create label for x-axis
    team_df['game_label'] = team_df['season'].astype(str) + " W" + team_df['week'].astype(str)

    # Plot with Plotly
    fig = px.line(team_df, x='game_label', y='rolling_stat',
                  title=f'{team_abbr} - {stat} (Rolling {window}-Game Average)',
                  labels={'rolling_stat': stat, 'game_label': 'Game'},
                  markers=True)

    fig.update_layout(xaxis_tickangle=45)
    fig.show()


df = pd.read_csv("complete_betting_dataset.csv")
"""
# Histograms of key features
df.hist(['point_diff_advantage', 'yards_advantage', 'home_team'], bins=20, figsize=(12, 6))

# Correlation heatmap
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
#plt.show(block=False)"""
plot_feature_correlations(df, target_col='home_win')
plot_team_trend(df, team_abbr='PHI', stat='home_score')
compare_teams(df, 'BUF', 'NYJ', stat='home_score')
important_features = ['home_score', 'away_score', 'home_point_diff', 'rolling_point_diff_advantage']
plot_logistic_feature_importance(df, important_features)
create_team_stat_trend(df, team_abbr='BUF', stat='home_point_diff', window=5)
plt.show()




#print(df.columns)
