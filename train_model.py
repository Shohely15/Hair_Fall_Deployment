import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =========================
# Load Dataset
# =========================

df = pd.read_csv("Predicting Hair Loss Severity.csv")

# =========================
# Remove SID if exists
# =========================

if 'SID' in df.columns:
    df.drop(columns=['SID'], inplace=True)

# =========================
# Handle Missing Values
# =========================

df.fillna(method='ffill', inplace=True)

# =========================
# Convert All Object Columns to String
# =========================

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str)

# =========================
# Encode All Text Columns
# =========================

label_encoders = {}

for col in df.columns:

    if df[col].dtype == 'object':

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        label_encoders[col] = le

# =========================
# Target
# =========================

TARGET = 'Current Hair Fall Severity'

X = df.drop(TARGET, axis=1)

y = df[TARGET]

# =========================
# Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Scale
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# =========================
# Train Model
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =========================
# Save Files
# =========================

pickle.dump(model, open("model.pkl", "wb"))

pickle.dump(scaler, open("scaler.pkl", "wb"))

pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

pickle.dump(X.columns.tolist(), open("feature_names.pkl", "wb"))

print("All files saved successfully!")
