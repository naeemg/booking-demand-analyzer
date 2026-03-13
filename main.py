import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def prepare_data():
    # Generating synthetic booking data for demonstration
    # In production, this would load from your SQL/CSV export
    np.random.seed(42)
    data = pd.DataFrame({
        'day_of_week': np.random.randint(0, 7, 100),
        'hour_of_day': np.random.randint(8, 20, 100),
        'is_holiday': np.random.randint(0, 2, 100),
        'prev_day_volume': np.random.randint(10, 50, 100),
        'booking_volume': np.random.randint(10, 60, 100)
    })
    return data

def train_model(df):
    X = df.drop('booking_volume', axis=1)
    y = df['booking_volume']
    
    # Split: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model: RandomForest is ideal for non-linear booking patterns
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Training Complete. MSE: {mse:.2f}")
    return model

if __name__ == "__main__":
    df = prepare_data()
    model = train_model(df)
