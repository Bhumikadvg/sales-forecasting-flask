import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# ✅ FIX the file path here
df = pd.read_csv('E:\\Sales\\Mall_Customers.csv')

# Replace these column names with actual ones from your CSV
X = df[['Annual Income (k$)']]   # Use actual column name from your data
y = df['Spending Score (1-100)'] # Target variable


model = LinearRegression()
model.fit(X, y)

# Save the model to model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")