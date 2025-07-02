import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# Load dataset
df = pd.read_csv("complete_betting_dataset.csv")

# Extract team and stat options
team_options = sorted(set(df['home_team'].dropna().unique()) | set(df['away_team'].dropna().unique()))
stat_options = [col for col in df.columns if col.startswith('home_') and df[col].dtype in ['float64', 'int64']]

# Start Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("NFL Team Stat Tracker", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Select Team:"),
        dcc.Dropdown(id='team-dropdown', options=[{'label': team, 'value': team} for team in team_options],
                     value='KC')
    ], style={'width': '40%', 'display': 'inline-block', 'padding': '10px'}),

    html.Div([
        html.Label("Select Stat:"),
        dcc.Dropdown(id='stat-dropdown', options=[{'label': stat, 'value': stat} for stat in stat_options],
                     value='home_score')
    ], style={'width': '40%', 'display': 'inline-block', 'padding': '10px'}),

    dcc.Graph(id='trend-plot')
])

@app.callback(
    Output('trend-plot', 'figure'),
    Input('team-dropdown', 'value'),
    Input('stat-dropdown', 'value')
)
def update_plot(team_abbr, stat):
    # Filter and prepare data
    home_df = df[df['home_team'] == team_abbr].copy()
    away_df = df[df['away_team'] == team_abbr].copy()

    # Adjust stat for away side
    away_stat = stat.replace('home_', 'away_')

    home_df['team_stat'] = home_df[stat]
    away_df['team_stat'] = away_df[away_stat]

    combined = pd.concat([home_df, away_df])
    combined = combined.sort_values(['season', 'week'])
    combined['rolling_avg'] = combined['team_stat'].rolling(5).mean()
    combined['game_label'] = combined['season'].astype(str) + " W" + combined['week'].astype(str)

    # Plot
    fig = px.line(combined, x='game_label', y='rolling_avg',
                  title=f'{team_abbr} - Rolling Avg of {stat} (5 Games)',
                  markers=True)
    fig.update_layout(xaxis_tickangle=45, xaxis_title='Game', yaxis_title=stat)
    return fig

if __name__ == '__main__':
    app.run(debug=True)
