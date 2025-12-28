"""
Cricket Score Predictor - Model Training Script
Trains an XGBoost regression model to predict T20 cricket scores.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(csv_path='t20i_info.csv'):
    """Load and prepare the cricket dataset."""
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Fill missing city values from venue
    df['city'] = df['city'].fillna(df['venue'].apply(lambda x: x.split(' ')[0]))
    
    # Select top 10 teams
    teams = [
        'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
        'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
    ]
    
    # Filter for selected teams
    df = df[df['batting_team'].isin(teams)]
    df = df[df['bowling_team'].isin(teams)]
    
    print(f"Dataset filtered. Shape: {df.shape}")
    return df


def engineer_features(df):
    """Create features for the prediction model."""
    print("Engineering features...")
    
    # Calculate cumulative scores per match
    df['current_score'] = df.groupby('match_id')['runs'].cumsum()
    
    # Calculate over and ball information
    df['over'] = df['ball'].apply(lambda x: int(x))
    df['ball_no'] = df['ball'].apply(lambda x: round((x - int(x)) * 10))
    df['ball_bowled'] = df['over'] * 6 + df['ball_no']
    df['balls_left'] = 120 - df['ball_bowled']
    
    # Calculate wickets
    df['player_dismissed'] = df['player_dismissed'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['wicket_left'] = 10 - df.groupby('match_id')['player_dismissed'].cumsum()
    
    # Calculate current run rate
    df['current_run_rate'] = (df['current_score'] * 6) / df['ball_bowled']
    df['current_run_rate'] = df['current_run_rate'].replace([np.inf, -np.inf], 0)
    
    # Calculate runs in last 5 overs
    df['last_five'] = df.groupby('match_id')['runs'].apply(
        lambda x: x.rolling(window=30, min_periods=1).sum()
    ).reset_index(level=0, drop=True)
    
    # Get total runs per match
    total_runs = df.groupby('match_id')['runs'].sum().reset_index()
    total_runs.columns = ['match_id', 'total_runs']
    
    # Merge total runs as target
    df = df.merge(total_runs, on='match_id')
    
    print(f"Features engineered. Shape: {df.shape}")
    return df


def prepare_final_dataset(df):
    """Prepare the final dataset for training."""
    print("Preparing final dataset...")
    
    # Select required columns
    final_df = df[['batting_team', 'bowling_team', 'city', 'current_score', 
                   'balls_left', 'wicket_left', 'current_run_rate', 
                   'last_five', 'total_runs']].copy()
    
    # Drop rows with missing values
    final_df.dropna(inplace=True)
    
    # Filter to only use data after 5 overs (ball_bowled > 30)
    # This is when last_five becomes meaningful
    final_df = final_df[final_df['last_five'] > 0]
    
    # Shuffle the data
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final dataset ready. Shape: {final_df.shape}")
    return final_df


def train_model(final_df):
    """Train the XGBoost model."""
    print("\nTraining model...")
    
    # Split features and target
    X = final_df.drop(columns=['total_runs'])
    y = final_df['total_runs']
    
    # Rename columns to match app.py expectations
    X.columns = ['batting_team', 'bowling_team', 'city', 'current_score', 
                 'balls_left', 'wickets_left', 'crr', 'last_five']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create preprocessing transformer
    transformer = ColumnTransformer([
        ('encoder', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
         ['batting_team', 'bowling_team', 'city'])
    ], remainder='passthrough')
    
    # Create the pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', transformer),
        ('scaler', StandardScaler()),
        ('regressor', XGBRegressor(
            n_estimators=1000,
            learning_rate=0.2,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Fit the model
    print("Fitting model (this may take a few minutes)...")
    pipe.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n✓ Model trained successfully!")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Mean Absolute Error: {mae:.2f} runs")
    
    return pipe


def save_model(pipe, filename='pipe.pkl'):
    """Save the trained model."""
    with open(filename, 'wb') as f:
        pickle.dump(pipe, f)
    print(f"\n✓ Model saved to {filename}")


def main():
    """Main training function."""
    print("=" * 50)
    print("Cricket Score Predictor - Model Training")
    print("=" * 50)
    
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Engineer features
        df = engineer_features(df)
        
        # Prepare final dataset
        final_df = prepare_final_dataset(df)
        
        # Train model
        pipe = train_model(final_df)
        
        # Save model
        save_model(pipe)
        
        print("\n" + "=" * 50)
        print("Training complete! You can now run the Flask app.")
        print("=" * 50)
        
    except FileNotFoundError:
        print("Error: t20i_info.csv not found!")
        print("Please ensure the dataset file is in the same directory.")
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == '__main__':
    main()
