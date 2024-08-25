import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from datetime import timedelta
import numpy as np


# Database connection parameters
conn_params = {
    'dbname': 'your_database',
    'user': 'your_user',
    'password': 'your_password',
    'host': 'localhost',
    'port': '5432'
}

def read_data():
    """Read data from PostgreSQL."""
    conn = psycopg2.connect(**conn_params)
    query = "SELECT * FROM intraday_data"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def pca_drift_detection(reference_data, current_data, variance_threshold=0.95, threshold=0.1):
    """
    Perform PCA on the reference data and check for drift in the current data.
    """
    pca = PCA(n_components=variance_threshold)
    pca.fit(reference_data.drop(columns=['close']))
    reference_components = pca.transform(reference_data.drop(columns=['close']))

    # Transform current data using PCA
    current_components = pca.transform(current_data.drop(columns=['close']))

    # Check for drift in the principal components
    drift_detected = np.any(
        np.abs(reference_components.mean(axis=0) - current_components.mean(axis=0)) > threshold
    )
    
    return drift_detected

def cusum_drift_detection(reference_data, current_data, threshold=0.5, drift_threshold=5):
    """
    Perform CUSUM-based drift detection on the current data compared to the reference data.
    """
    s_positive = 0
    s_negative = 0
    
    # Calculate the mean of the reference data
    reference_mean = reference_data['close'].mean()

    # Track cumulative sum for the current data
    for i, row in current_data.iterrows():
        error = row['close'] - reference_mean
        
        # Update the cumulative sums
        s_positive = max(0, s_positive + error - threshold)
        s_negative = max(0, s_negative - error - threshold)

        # Check if either sum exceeds the drift threshold
        if s_positive > drift_threshold or s_negative > drift_threshold:
            return True

    return False
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def linear_regression_drift_detection(reference_data, current_data, error_threshold=0.1, test_size=0.2, random_state=None):
    """
    Train a linear regression model on a split of the reference data and predict on the current data.
    Detect concept drift based on prediction error.
    
    Parameters:
    - reference_data: pd.DataFrame - The historical data used to train the model.
    - current_data: pd.DataFrame - The new data on which to predict and detect drift.
    - error_threshold: float - The threshold above which drift is detected.
    - test_size: float - The proportion of the reference data to include in the test split.
    - random_state: int or None - Random seed for reproducibility of the train-test split.
    
    Returns:
    - drift_detected: bool - True if drift is detected, False otherwise.
    """
    # Define the target column
    target_column = 'close'
    
    # Split the reference data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        reference_data.drop(columns=[target_column]), 
        reference_data[target_column], 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Train the linear regression model on the training data
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Optionally, evaluate the model on the test data (for internal validation)
    test_predictions = model.predict(X_test)
    test_error = mean_absolute_error(y_test, test_predictions)
    print(f"Mean Absolute Error on reference test set: {test_error}")
    
    # Predict the 'close' values for the current data using its features
    predictions = model.predict(current_data.drop(columns=[target_column]))
    
    # Calculate the mean absolute error between the actual and predicted 'close' values for the current data
    error = mean_absolute_error(current_data[target_column], predictions)
    print(f"Mean Absolute Error on current data: {error}")
    
    # Check if the error exceeds the threshold to detect drift
    drift_detected = error > error_threshold
    
    return drift_detected

if __name__ == "__main__":
    # Read the data from the database
    df = read_data()
    print(df.head())
    
    # Ensure 'datetime' is a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Get the latest date in the DataFrame
    latest_date = df.index.max()
    
    # Filter the DataFrame to get the latest day's data
    current = df.loc[latest_date.strftime('%Y-%m-%d')]
    
    # Define the reference date range (e.g., last 365 days)
    reference_start_date = latest_date - timedelta(days=365)
    reference_end_date = latest_date - timedelta(days=1)
    
    # Filter the DataFrame to get the reference data
    reference = df.loc[reference_start_date:reference_end_date]


    drift_detected=linear_regression_drift_detection(reference, current)
    if drift_detected:
        print(f"Concept drift detected on {latest_date.strftime('%Y-%m-%d')}, retraining the model...")

        # Train a new model using the combined reference and current data
        new_data = pd.concat([reference, current])
        model = RandomForestRegressor()
        model.fit(new_data.drop(columns=['close']), new_data['close'])
        
        # Save and deploy the new model
        joblib.dump(model, 'stock_model1_updated.pkl')

    


    # # Check for concept drift using PCA-based detection
    # drift_detected = pca_drift_detection(reference, current)

    # if drift_detected:
    #     print(f"Concept drift detected on {latest_date.strftime('%Y-%m-%d')}, retraining the model...")

    #     # Train a new model using the combined reference and current data
    #     new_data = pd.concat([reference, current])
    #     model = RandomForestRegressor()
    #     model.fit(new_data.drop(columns=['close']), new_data['close'])
        
    #     # Save and deploy the new model
    #     joblib.dump(model, 'stock_model_updated.pkl')
    # else:
    #     print("No concept drift detected. No retraining necessary.")
    
    
    
    #Check for concept drift using CUSUM-based detection
    #drift_detected = cusum_drift_detection(reference, current)

    # if drift_detected:
    #     print(f"Concept drift detected on {latest_date.strftime('%Y-%m-%d')}, retraining the model...")

    #     # Train a new model using the combined reference and current data
    #     new_data = pd.concat([reference, current])
    #     model = RandomForestRegressor()
    #     model.fit(new_data.drop(columns=['close']), new_data['close'])
        
    #     # Save and deploy the new model
    #     joblib.dump(model, 'stock_model_updated.pkl')
    # else:
    #     print("No concept drift detected. No retraining necessary.")

