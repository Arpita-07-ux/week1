#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Importing Data
crop=pd.read_csv(r"C:\Users\arpit\Downloads\Crop_recommendation.csv")
print(crop.head())
print(crop.shape)
print(crop.info())
print(crop.isnull().sum())
print(crop.duplicated().sum())
print(crop['label'].value_counts())


#Encoding
crop_dict={
    'rice':1,
    'maize':2,
    'chickpea':3,
    'kidneybeans':4,
    'pigeonpeas':5,
    'mothbeans':6,
    'mungbean':7,
    'blackgram': 8,
    'lentil':9,
    'pomegranate':10,
    'banana':11,
    'mango':12,
    'grapes':13,
    'watermelon':14,
    'muskmelon':15,
    'apple':16,
    'orange':17,
    'papaya':18,
    'coconut':19,
    'cotton':20,
    'jute':21,
    'coffee':22,
}
crop['crop_num']=crop['label'].map(crop_dict)
# print(crop['crop_num'].value_counts())
crop.drop(['label'],axis=1,inplace=True)


#train test split
x=crop.drop('crop_num',axis=1)
y=crop['crop_num']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


#scale the feature using MinMaxScaler
ms=MinMaxScaler()
x_train=ms.fit_transform(x_train)
x_test=ms.transform(x_test)

#training model
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
ypred=rfc.predict(x_test)
print(accuracy_score(y_test,ypred))


#predictive system
def recommendation(N,P,K,temperature,humidity,ph,rainfall):
    features=np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    transformed_features=ms.fit_transform(features)
    prediction=rfc.predict(transformed_features)
    return prediction[0]

N=40
P=50
K=50
temperature=40.0
humidity=20
ph=100
rainfall=100

predict=recommendation(N,P,K,temperature,humidity,ph,rainfall)

crop_dict={
    1:'rice',
    2:'maize',3:'chickpea',4:'kidneybeans',5:'pigeonpeas',
    6:'mothbeans',7:'mungbean',8:'blackgram',9:'lentil',
    10:'pomegranate',11:'banana',12:'mango',13:'grapes',
    14:'watermelon',15:'muskmelon',16:'apple',17:'orange',
    18:'papaya',19:'coconut',20:'cotton',21:'jute',22:'coffee'
}

if predict in crop_dict:
   crop = crop_dict[predict]
   print(f"{crop} is the best crop to be cultivated.")
else:
    print("Sorry ,We are not able to reccomend a proper crop for this environment.")
