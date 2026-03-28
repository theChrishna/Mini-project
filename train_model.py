import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the dataset you just created
header_names = (
    ["label"] + [f"pt{i}_x" for i in range(21)] + [f"pt{i}_y" for i in range(21)]
)
df = pd.read_csv("hand_data.csv", header=None, names=header_names)

# 2. Separate the 'label' from the 'features'
X = df.drop("label", axis=1)
y = df["label"]

# 3. Split into training (80%) and testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Check accuracy
y_predict = model.predict(X_test)
score = accuracy_score(y_test, y_predict)
print(f"{score * 100:.2f}% of test samples were classified correctly!")

# 6. Save the model to a "Pickle" file
with open("model.p", "wb") as f:
    pickle.dump(model, f)
