import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import warnings

# Ignore UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset
df=pd.read_csv("diabetes_prediction_dataset.csv")

# Check for missing values
print("Missing values in the dataset:")
print(df.isnull().sum())

# Drop rows with missing values
df=df.dropna()

# Preprocessing using Ordinal Encoder
enc=OrdinalEncoder()
df["smoking_history"]=enc.fit_transform(df[["smoking_history"]])
df["gender"]=enc.fit_transform(df[["gender"]])

# Define Independent and Dependent Variables
x=df.drop("diabetes", axis=1)
y=df["diabetes"]

# Check data types
print("\nData types of features:")
print(x.dtypes)

# 70% data - Train and 30% data - Test
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=42)

# Check shapes
print("\nShapes of training and testing data:")
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

# RandomForest Algorithm
model=RandomForestClassifier(random_state=42).fit(x_train, y_train)
y_pred=model.predict(x_test)
accuracy=metrics.accuracy_score(y_test, y_pred)

# Print accuracy
print(f"\nModel Accuracy: {accuracy:.2f}")

# Function to get user input and make predictions
def predict_diabetes():
    print("\nEnter the following details to predict diabetes:")
    gender=input("Gender (Male/Female/Other): ")
    age=input("Age: ")
    hypertension=input("Hypertension (No/Yes): ")
    heart_disease=input("Heart Disease (No/Yes): ")
    smoking_history=input("Smoking History (Never/Current/Former/Ever/Not Current/No Info): ")
    bmi=input("BMI: ")
    hba1c_level=input("HbA1c Level: ")
    blood_glucose_level=input("Blood Glucose Level: ")

    # Dictionaries for encoding user input
    gender_dict={'Female': 0.0, 'Male': 1.0, 'Other': 2.0}
    hypertension_dict={'No': 0, 'Yes': 1}
    heart_disease_dict={'No': 0, 'Yes': 1}
    smoking_history_dict={'Never': 4.0, 'No Info': 0.0, 'Current': 1.0,
                            'Former': 3.0, 'Ever': 2.0, 'Not Current': 5.0}

    try:
        # Convert user input to a numpy array
        user_data=np.array([[gender_dict[gender], float(age), hypertension_dict[hypertension],
                               heart_disease_dict[heart_disease], smoking_history_dict[smoking_history],
                               float(bmi), float(hba1c_level), float(blood_glucose_level)]])

        # Make prediction
        test_result=model.predict(user_data)

        # Display result
        if test_result[0] == 0:
            print("\nDiabetes Result: Negative")
        else:
            print("\nDiabetes Result: Positive (Please Consult with Doctor)")

    except Exception as e:
        print(f"\nAn error occurred: {e}. Please check your input and try again.")

# Run the prediction function
predict_diabetes()