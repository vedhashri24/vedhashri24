# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 2: Load dataset (download 'train.csv' from Kaggle House Prices competition)
data = pd.read_csv("train.csv")

# Step 3: Basic preprocessing
# Drop rows with missing target
data = data.dropna(subset=["SalePrice"])

# Select a few useful numeric features
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
X = data[features]
y = data["SalePrice"]

# Step 4: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a smart regression model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions and evaluate
predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)

print(f"Root Mean Squared Error: ${rmse:,.2f}")
