import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
crop=pd.read_csv(r"C:\Users\arpit\Downloads\Crop_recommendation.csv")
print(crop.head())
print(crop.shape)
print(crop.info())
print(crop.isnull().sum())
print(crop.duplicated().sum())
x=crop.drop('label',axis=1)
y=crop['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("done")
