import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib

# Load the data
df_matches = pd.read_csv('matches.csv')
df_standings = pd.read_csv('standings.csv')

# Data preprocessing and feature engineering
completed_matches = df_matches.loc[df_matches['status'] == 'FINISHED'].copy()
completed_matches.dropna(subset=['home_score', 'away_score'], inplace=True)

def determine_result(row):
    if row['home_score'] > row['away_score']:
        return 1  # Home Win
    elif row['home_score'] < row['away_score']:
        return -1  # Away Win
    else:
        return 0  # Draw

completed_matches['result'] = completed_matches.apply(determine_result, axis=1)
completed_matches['goal_difference'] = completed_matches['home_score'] - completed_matches['away_score']
completed_matches['home_advantage'] = 1

# Merge home and away team standings
completed_matches = completed_matches.merge(
    df_standings.add_prefix('home_'),
    left_on='home_team', right_on='home_team_name', how='left'
)

completed_matches = completed_matches.merge(
    df_standings.add_prefix('away_'),
    left_on='away_team', right_on='away_team_name', how='left'
)

# Normalise numerical features
scaler = StandardScaler()
numerical_features = ['goal_difference', 'home_position', 'away_position', 'home_points', 'away_points']
completed_matches[numerical_features] = scaler.fit_transform(completed_matches[numerical_features])

# Define features and target
features = ['goal_difference', 'home_position', 'away_position', 'home_points', 'away_points', 'home_advantage']
X = completed_matches[features]
y = completed_matches['result']

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'match_outcome_predictor.pkl')

# Evaluate the model
y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
