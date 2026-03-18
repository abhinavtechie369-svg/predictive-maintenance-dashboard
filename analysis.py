import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# load dataset
data = pd.read_csv("data/ai4i2020.csv")

print(data.head())
print(data.info())
print(data.describe())

# check missing values
print(data.isnull().sum())

# MACHINE FAILURE DISTRIBUTION
plt.figure(figsize=(6,4))
sns.countplot(x="Machine failure", data=data)
plt.title("Machine Failure Distribution")
plt.show()

# CORRELATION HEATMAP

plt.figure(figsize=(10,6))
sns.heatmap(data.corr(numeric_only=True), annot=True)

plt.title("Sensor Correlation Heatmap")

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# FEATURES
X = data[[
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]]

# TARGET
y = data["Machine failure"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL
model = RandomForestClassifier()

# TRAIN
model.fit(X_train, y_train)

# PREDICTIONS
predictions = model.predict(X_test)

# ACCURACY
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print("Confusion Matrix:")
print(cm)