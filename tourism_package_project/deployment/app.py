import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(
    repo_id="23gaurav-verma/tourism-package-predict-model",
    filename="best_tourism_package_prediction_model_v1.joblib"
)

model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction")

st.write("""
Predict whether a customer is likely to purchase the newly introduced
**Wellness Tourism Package** based on customer demographics and interaction details.
""")

# User input
Age = st.number_input("Age", 18, 70, 35)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Enquiry"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

DurationOfPitch = st.number_input(
    "Duration of Pitch (minutes)",
    5,
    130,
    15
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

Gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    1,
    10,
    2
)

NumberOfFollowups = st.number_input(
    "Number of Follow-ups",
    1,
    10,
    3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

NumberOfTrips = st.number_input(
    "Number of Trips",
    1,
    25,
    3
)

Passport = st.selectbox(
    "Passport",
    [0, 1]
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1]
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    0,
    5,
    1
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    1000,
    100000,
    25000
)

# Model label encoding
TypeofContact = {
    "Company Invited": 0,
    "Self Enquiry": 1
}[TypeofContact]

Occupation = {
    "Free Lancer": 0,
    "Large Business": 1,
    "Salaried": 2,
    "Small Business": 3
}[Occupation]

Gender = {
    "Female": 0,
    "Male": 1
}[Gender]

ProductPitched = {
    "Basic": 0,
    "Deluxe": 1,
    "King": 2,
    "Standard": 3,
    "Super Deluxe": 4
}[ProductPitched]

MaritalStatus = {
    "Divorced": 0,
    "Married": 1,
    "Single": 2
}[MaritalStatus]

Designation = {
    "AVP": 0,
    "Executive": 1,
    "Manager": 2,
    "Senior Manager": 3,
    "VP": 4
}[Designation]

# Assemble input into DataFrame
input_data = pd.DataFrame([{

    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome

}])


if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:

        st.success("Customer is likely to purchase the Wellness Tourism Package.")

    else:

        st.error("Customer is unlikely to purchase the Wellness Tourism Package.")
