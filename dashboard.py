import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load pre-trained model
loaded_model = joblib.load('match_outcome_predictor.pkl')

# Load data
matches = pd.read_csv('matches.csv')
standings = pd.read_csv('standings.csv')

# Define the features for prediction
features = [
    'goal_difference', 'home_position', 'away_position',
    'home_points', 'away_points', 'home_advantage'
]

# Title and Navigation
st.title("Football Match Prediction Dashboard")
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to:", ['Overview', 'Match Predictions', 'Team Statistics'])

# Prepare data for upcoming matches (common to both 'Match Predictions' and 'Head-to-Head' functionality)
upcoming_matches = matches[matches['status'] != 'FINISHED']
upcoming_matches = upcoming_matches.merge(
    standings[['team_name', 'goal_difference', 'position', 'points']],
    left_on='home_team', right_on='team_name', how='left'
).rename(columns={
    'goal_difference': 'home_goal_difference',
    'position': 'home_position',
    'points': 'home_points'
})

upcoming_matches = upcoming_matches.merge(
    standings[['team_name', 'goal_difference', 'position', 'points']],
    left_on='away_team', right_on='team_name', how='left'
).rename(columns={
    'goal_difference': 'away_goal_difference',
    'position': 'away_position',
    'points': 'away_points'
})

upcoming_matches['home_advantage'] = 1
upcoming_matches['goal_difference'] = (
    upcoming_matches['home_goal_difference'] - upcoming_matches['away_goal_difference']
)
upcoming_features = upcoming_matches[features]

# Predict match outcomes 
upcoming_matches['predicted_result'] = loaded_model.predict(upcoming_features)

# Title and Navigation Logic
if options == 'Overview':
    st.header("League Overview")
    st.write("League Standings")
    st.dataframe(standings)

elif options == 'Match Predictions':
    st.header("Upcoming Match Predictions")
    st.write("Predicted Results for Upcoming Matches")

    # Display predictions
    st.dataframe(upcoming_matches[['home_team', 'away_team', 'predicted_result']])

elif options == 'Team Statistics':
    st.header("Team Statistics")
    st.write("Team Performance Over Time")
    fig = px.bar(standings, x='team_name', y='points', color='team_name', title="Current Points by Team")
    st.plotly_chart(fig)

# Team selection for prediction outcome
home_team = st.selectbox("Select Home Team", matches['home_team'].unique())
away_team = st.selectbox("Select Away Team", matches['away_team'].unique())

# Update the prediction based on selected teams
selected_match = upcoming_matches[(upcoming_matches['home_team'] == home_team) & (upcoming_matches['away_team'] == away_team)]
if not selected_match.empty:
    st.write("Predicted Outcome:", selected_match['predicted_result'].values[0])
else:
    st.write("No upcoming match found for selected teams.")

# Adding prediction probabilities
predicted_probs = loaded_model.predict_proba(upcoming_features)
upcoming_matches['home_win_prob'] = predicted_probs[:, 0]
upcoming_matches['draw_prob'] = predicted_probs[:, 1]
upcoming_matches['away_win_prob'] = predicted_probs[:, 2]

st.dataframe(upcoming_matches[['home_team', 'away_team', 'home_win_prob', 'draw_prob', 'away_win_prob']])

# Head-to-head comparison
team_1 = st.selectbox("Select First Team", matches['home_team'].unique())
team_2 = st.selectbox("Select Second Team", matches['away_team'].unique())

# Filter for past results between these two teams
head_to_head = matches[((matches['home_team'] == team_1) & (matches['away_team'] == team_2)) |
                       ((matches['home_team'] == team_2) & (matches['away_team'] == team_1))]
st.write("Head-to-Head Results:", head_to_head[['home_team', 'away_team', 'home_score', 'away_score']])

# Retrain Model Button
if st.button('Retrain Model'):
    
    X_train = upcoming_matches[features] 
    y_train = upcoming_matches['predicted_result']  

    # Retrain the loaded model
    loaded_model.fit(X_train, y_train)
    
    # Save the retrained model
    joblib.dump(loaded_model, 'match_outcome_predictor.pkl')
    
    st.write("Model retrained and saved!")


# Sort and display predictions
st.dataframe(upcoming_matches[['home_team', 'away_team', 'predicted_result']].sort_values('predicted_result'))
