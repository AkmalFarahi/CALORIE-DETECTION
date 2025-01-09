import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Read dataset
df = pd.read_csv('C:\\Users\\user\\Downloads\\CalorieDetectionProject-main\\Dataset\\bmi_data.csv')

# Renaming columns for easier handling
df.rename(columns={'Height(Inches)': 'Height', 'Weight(Pounds)': 'Weight'}, inplace=True)

# Handling missing values
df.dropna(inplace=True)

# Encode categorical columns (e.g., Gender, Activity Level)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Male=0, Female=1
df['Activity_Level'] = label_encoder.fit_transform(df['Activity_Level'])  # Sedentary=0, Active=1, Highly Active=2

# Create BMI column for the prediction task (this is just for illustration, you may already have it)
df['BMI'] = df['Weight'] / (df['Height']**2) * 703

# Select Features and Target
X = df[['Height', 'Weight', 'Age', 'Gender', 'Activity_Level']]  # Features
y = df['BMI']  # Target: BMI

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model (for example: Linear Regression to predict BMI)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Save the model
joblib.dump(regressor, 'bmi_predictor_model.pkl')

# Model Evaluation (if needed)
print(f"Model accuracy: {regressor.score(X_test, y_test)}")
