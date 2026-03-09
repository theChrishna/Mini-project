import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# 1. Load data (no header)
df = pd.read_csv('hand_data.csv', header=None)

X = df.iloc[:, 1:] # All coordinate columns
y = df.iloc[:, 0]  # The Label column

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. Save the fresh model
with open('model.p', 'wb') as f:
    pickle.dump(model, f)

# 5. Diagnostic Report
y_pred = model.predict(X_test)
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix (Should be mostly diagonal) ---")
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)