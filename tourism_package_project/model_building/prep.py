# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOURISM_TOKEN"))
DATASET_PATH = "hf://datasets/23gaurav-verma/tourism-package-predict-dataset/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ---------------------------------------------------------
# Drop unnecessary columns
# ---------------------------------------------------------

df.drop(columns=["CustomerID", "Unnamed: 0"], inplace=True)

# ---------------------------------------------------------
# Fix inconsistent values
# ---------------------------------------------------------

# Gender
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

# Marital Status
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

# ---------------------------------------------------------
# Encode categorical columns
# ---------------------------------------------------------

categorical_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Categorical columns encoded successfully.")

# ---------------------------------------------------------
# Target column
# ---------------------------------------------------------

target_col = "ProdTaken"

# Spliting into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# ---------------------------------------------------------
# Train-Test Split
# ---------------------------------------------------------

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# Save locally
# ---------------------------------------------------------

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train-Test split saved successfully.")

# ---------------------------------------------------------
# Upload to Hugging Face Dataset
# ---------------------------------------------------------

files = [
    "Xtrain.csv",
    "Xtest.csv",
    "ytrain.csv",
    "ytest.csv"
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="23gaurav-verma/tourism-package-predict-dataset",   # <-- Update if different
        repo_type="dataset"
    )

print("Files uploaded successfully.")
