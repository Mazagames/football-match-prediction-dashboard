# Football Match Prediction Dashboard

This project uses machine learning to predict the outcomes of football matches based on team statistics, previous performance, and other factors. The predictions are made using a trained model that takes into account key features such as team positions, goal differences, points, and home advantage.

The dashboard is built using **Streamlit** for visualizing predictions and providing an interactive interface.

## Files

- **`app.py`**: The main Streamlit app file that serves the dashboard interface.
- **`match_outcome_predictor.pkl`**: Pre-trained machine learning model used for predicting match outcomes.
- **`matches.csv`**: Matches data file (fetched from the API).
- **`standings.csv`**: Standings data file (fetched from the API).
- **`data_fetcher.py`**: Script to fetch match and standings data from the football-data.org API.
- **`model_training.py`**: Script to preprocess data, train the machine learning model, and save it to a `.pkl` file.
- **`requirements.txt`**: List of Python dependencies required to run the project.
- **`README.md`**: Project documentation.

## Setup Instructions

To run this project on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bamg2/football-match-prediction-dashboard.git
