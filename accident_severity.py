import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Data/test.csv")

print("Original dataset shape:", df.shape)

# Remove unnecessary ID column
df = df.drop(columns=["Num"])

# Find categorical columns
categorical_columns = df.select_dtypes(include=["object", "str"]).columns

# Fill missing categorical values
df[categorical_columns] = df[categorical_columns].fillna("Unknown")

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned dataset shape:", df.shape)

# Remove records with unknown casualty severity
df = df[df["Casualty_severity"] != "na"]

print("\nDataset after removing unknown severity:")
print(df.shape)

print("\nCasualty severity distribution:")
print(df["Casualty_severity"].value_counts())


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Casualty_severity"
)

plt.title("Distribution of Casualty Severity")
plt.xlabel("Casualty Severity")
plt.ylabel("Number of Cases")

plt.tight_layout()
plt.savefig("Images/severity_distribution.png")
plt.show()

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Weather_conditions",
    hue="Casualty_severity"
)

plt.title("Casualty Severity by Weather Conditions")
plt.xlabel("Weather Conditions")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("Images/weather_vs_severity.png")
plt.show()



plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Light_conditions",
    hue="Casualty_severity"
)

plt.title("Casualty Severity by Light Conditions")
plt.xlabel("Light Conditions")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("Images/light_vs_severity.png")
plt.show()



plt.figure(figsize=(12, 7))

sns.countplot(
    data=df,
    y="Cause_of_accident",
    hue="Casualty_severity"
)

plt.title("Casualty Severity by Cause of Accident")
plt.xlabel("Number of Cases")
plt.ylabel("Cause of Accident")

plt.tight_layout()
plt.savefig("Images/cause_vs_severity.png")
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Select a small number of meaningful features
features = [
    "Weather_conditions",
    "Light_conditions",
    "Road_surface_conditions",
    "Driving_experience",
    "Type_of_collision",
    "Cause_of_accident",
    "Number_of_vehicles_involved",
    "Number_of_casualties"
]

# Create input data and target
X = df[features]
y = df["Casualty_severity"]

# Convert categorical columns into numbers
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create the Decision Tree model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nDecision Tree Accuracy:", accuracy)

# Show detailed results
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Show confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


import matplotlib.pyplot as plt

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
importance.sort_values().plot(kind="barh")

plt.title("Top Features Used by the Decision Tree")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("Images/feature_importance.png")
plt.show()
