# train_model.py

import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# Sample dummy data
data = pd.DataFrame({
    'Origin': ['JFK', 'LAX', 'SFO', 'JFK', 'LAX'],
    'Destination': ['SFO', 'JFK', 'LAX', 'LAX', 'SFO'],
    'DepTime': [930, 1400, 1800, 1230, 600],
    'Incident': [0, 1, 0, 0, 1]
})

# Encode categorical features
X = data[['Origin', 'Destination', 'DepTime']]
y = data['Incident']

encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# Train model
model = HistGradientBoostingClassifier()
model.fit(X_encoded, y)

# Save model and encoder
joblib.dump(model, "/workspaces/Madesh9-aviation_final_project/models/incident_model.pkl")
joblib.dump(encoder, "/workspaces/Madesh9-aviation_final_project/models/encoder.pkl")
print("Model and encoder saved.")