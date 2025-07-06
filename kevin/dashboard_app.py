from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re

# Load data
main_df = pd.read_csv("complete_betting_dataset.csv")
test_df = pd.read_csv("logistic_regression_predictions.csv")
probs = test_df['win_probability'].values
outcomes = test_df['actual_outcome'].values

# Create list of stat options
stat_options = [col for col in main_df.columns if col.startswith('home_') and main_df[col].dtype in ['float64', 'int64'] and len(col) > 7]
teams = sorted(set(main_df['home_team'].dropna().unique()) | set(main_df['away_team'].dropna().unique()))

app = Dash(__name__)
app.title = "NFL Betting Analysis Dashboard"

app.layout = html.Div([
    html.H1("NFL Betting Analysis Dashboard", style={'textAlign': 'center'}),

    dcc.Tabs([
        dcc.Tab(label='Team Stat Trend vs. League Average', children=[
            html.Div([
                html.Label("Select Team:"),
                dcc.Dropdown(id='trend-team', options=[{'label': t, 'value': t} for t in teams], value='KC'),
                html.Label("Select Stat:"),
                dcc.Dropdown(id='trend-stat', options=[{'label': s, 'value': s} for s in stat_options], value='home_score'),
                dcc.Graph(id='team-trend-graph')
            ], style={'padding': '20px'})
        ]),

        dcc.Tab(label='Team Comparison (All Stats)', children=[
            html.Div([
                html.Label("Select Team 1:"),
                dcc.Dropdown(id='team1', options=[{'label': t, 'value': t} for t in teams], value='KC'),
                html.Label("Select Team 2:"),
                dcc.Dropdown(id='team2', options=[{'label': t, 'value': t} for t in teams], value='PHI'),
                dcc.Graph(id='comparison-graph')
            ], style={'padding': '20px'})
        ]),

        dcc.Tab(label='ROI Strategies (Test Set)', children=[
            html.Div([
                html.Label("Kelly Threshold"),
                dcc.Slider(min=0.5, max=0.9, step=0.01, value=0.6, id='kelly-threshold', marks=None),
                html.Label("Fixed Threshold"),
                dcc.Slider(min=0.5, max=0.9, step=0.01, value=0.55, id='fixed-threshold', marks=None),
                html.Label("Starting Bankroll ($)"),
                dcc.Input(id='bankroll', type='number', value=1000, step=100),
                dcc.Graph(id='roi-graph'),
                html.Div(id='summary-output', style={'textAlign': 'center'})
            ], style={'padding': '20px'})
        ])
    ])
])

# === Callbacks ===
@app.callback(
    Output('team-trend-graph', 'figure'),
    Input('trend-team', 'value'),
    Input('trend-stat', 'value')
)
def update_team_trend(team_abbr, stat):
    window = 5
    home_df = main_df[main_df['home_team'] == team_abbr].copy()
    away_df = main_df[main_df['away_team'] == team_abbr].copy()
    away_stat = stat.replace('home_', 'away_')
    home_df['team_stat'] = home_df[stat]
    away_df['team_stat'] = away_df[away_stat]
    combined = pd.concat([home_df, away_df]).sort_values(['season', 'week'])
    combined['rolling_stat'] = combined['team_stat'].rolling(window).mean()
    combined['game'] = combined['season'].astype(str) + ' W' + combined['week'].astype(str)

    main_df['avg_stat'] = main_df[[stat, away_stat]].mean(axis=1)
    main_df['league_rolling'] = main_df['avg_stat'].rolling(window).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined['game'], y=combined['rolling_stat'], mode='lines', name=f'{team_abbr} Avg'))
    fig.add_trace(go.Scatter(x=main_df['season'].astype(str) + ' W' + main_df['week'].astype(str),
                             y=main_df['league_rolling'], mode='lines', name='NFL Avg'))
    fig.update_layout(title=f"{team_abbr} vs NFL Average: {stat}", xaxis_title="Game", yaxis_title=stat)
    return fig

@app.callback(
    Output('comparison-graph', 'figure'),
    Input('team1', 'value'),
    Input('team2', 'value')
)
def update_team_comparison(team1, team2):
    stat_cols = [col for col in main_df.columns if col.startswith('home_') and main_df[col].dtype in [int, float] and col not in ['home_win']]
    team1_vals, team2_vals = [], []

    for col in stat_cols:
        away_col = col.replace('home_', 'away_')
        t1_home = main_df[main_df['home_team'] == team1][col].mean()
        t1_away = main_df[main_df['away_team'] == team1][away_col].mean()
        t2_home = main_df[main_df['home_team'] == team2][col].mean()
        t2_away = main_df[main_df['away_team'] == team2][away_col].mean()

        team1_vals.append(np.nanmean([t1_home, t1_away]))
        team2_vals.append(np.nanmean([t2_home, t2_away]))

    categories = [col.replace('home_', '') for col in stat_cols]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=team1_vals, name=team1))
    fig.add_trace(go.Bar(x=categories, y=team2_vals, name=team2))
    fig.update_layout(barmode='group', xaxis_tickangle=45, title=f"Team Comparison: {team1} vs {team2}")
    return fig

@app.callback(
    Output('roi-graph', 'figure'),
    Output('summary-output', 'children'),
    Input('kelly-threshold', 'value'),
    Input('fixed-threshold', 'value'),
    Input('bankroll', 'value')
)
def update_roi_plot(kelly_thresh, fixed_thresh, bankroll):
    def simulate_roi(probs, outcomes, method, threshold, bankroll):
        odds = 1.91
        confident_mask = (probs > threshold) | (probs < (1 - threshold))
        confident_probs = probs[confident_mask]
        confident_outcomes = outcomes[confident_mask]
        bets = []
        current_bankroll = bankroll
        for p, actual in zip(confident_probs, confident_outcomes):
            p_win = p if p > 0.5 else 1 - p
            predicted_win = int(p >= 0.5)
            stake = 10 if method == 'fixed' else max(0, min(((odds * p_win - (1 - p_win)) / (odds - 1)), 1)) * current_bankroll
            correct = predicted_win == actual
            profit = stake * (odds - 1) if correct else -stake
            current_bankroll += profit
            bets.append(current_bankroll)
        return bets

    kelly_bankrolls = simulate_roi(probs, outcomes, 'kelly', kelly_thresh, bankroll)
    fixed_bankrolls = simulate_roi(probs, outcomes, 'fixed', fixed_thresh, bankroll)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=fixed_bankrolls, mode='lines', name='Fixed Bet'))
    fig.add_trace(go.Scatter(y=kelly_bankrolls, mode='lines', name='Kelly Criterion'))
    fig.add_hline(y=bankroll, line_dash='dot', line_color='gray')
    fig.update_layout(title='ROI: Fixed vs Kelly (Test Set)', xaxis_title='Game', yaxis_title='Bankroll ($)')

    fixed_roi = (fixed_bankrolls[-1] - bankroll) / bankroll * 100 if fixed_bankrolls else 0
    kelly_roi = (kelly_bankrolls[-1] - bankroll) / bankroll * 100 if kelly_bankrolls else 0
    return fig, f"Fixed ROI: {fixed_roi:.2f}% | Kelly ROI: {kelly_roi:.2f}%"

if __name__ == '__main__':
    app.run(debug=True)
