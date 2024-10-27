import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv(r'C:\Users\shiva\ML\spam\data\spam.csv',encoding = "ISO-8859-1")

df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

#Label Encoding
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df.v1 = labelencoder.fit_transform(df.v1)

#Assigning value for X and Y
X =df['v2']
Y =df['v1']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=2,test_size=0.2)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english')
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

model = LogisticRegression()
model.fit(X_train_features,Y_train)

prediction = model.predict(X_train_features)
accuracy =accuracy_score(Y_train,prediction)

print('Train dataset accuracy : ',accuracy)

#Creating a pipeline
pipe=Pipeline(
    [
        ('vec',TfidfVectorizer()),
        ('clf',LogisticRegression()),
    ]
)

pipe.fit(X_train,Y_train)

#saving the pipe
joblib.dump(pipe,'modelpipe.joblib')


print("pandas==",pd.__version__)
print("numpy==",np.__version__)
print("sklearn==",sklearn.__version__)
print("joblib==",joblib.__version__)