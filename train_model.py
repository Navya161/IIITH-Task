import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Dummy dataset (replace later with real dataset)
data = pd.DataFrame({
    "distance": np.random.randint(50, 100, 200),
    "diff": np.random.randint(1, 10, 200),
    "slope": np.random.randint(0, 5, 200),
    "label": np.random.choice([0, 1], 200)
})

X = data[["distance", "diff", "slope"]]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("best_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved!")