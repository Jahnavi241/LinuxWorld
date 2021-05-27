# load dataset

import pandas as pd 
db=pd.read_csv("Salary_Data.csv")
print("Dataset Loaded Sucessfully.")

# Creating features and target.
#x = feature/independant variable
#y = target / dependant variable
x=db["YearsExperience"]
y=db["Salary"]

# loading LinearRegression
from sklearn.linear_model import LinearRegression
model=LinearRegression()
x = db["YearsExperience"].values.reshape(30,1)
model.fit(x,y)
print("Model created Sucessfully.")

# Saving trained model
import joblib
joblib.dump(model,'Salary_model.pkl')
print("MODEL SAVED SUCCESSFULLY.")
