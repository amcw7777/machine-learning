from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
import numpy as np
from sklearn import linear_model

def set_missing_ages(df):

    age_df = df[ ['Age','Fare','Parch','SibSp','Pclass']]

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    y = known_age[:,0]
    X = known_age[:,1:]

    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)

    predictedAges = rfr.predict(unknown_age[:,1::])

    df.loc[ (df.Age.isnull()), 'Age'] = predictedAges

    return df,rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()),'Cabin'] = "Yes"
    df.loc[ (df.Cabin.isnull()),'Cabin']  = "No"
    return df

def feature_factorization(df):
    dummies_Cabin = pd.get_dummies(df['Cabin'],prefix='Cabin')
    dummies_Embarked = pd.get_dummies(df['Embarked'],prefix='Embarked')
    dummies_Sex = pd.get_dummies(df['Sex'],prefix='Sex')
    dummies_Pclass = pd.get_dummies(df['Pclass'],prefix='Pclass')

    df = pd.concat( [df,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)
    df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)
    return df

def set_scale(df):
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df['Age'])
    df['Age_scaled'] = scaler.fit_transform(df['Age'],age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'])
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'],fare_scale_param)
    return df

def train(df):
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()
    y = train_np[:,0]
    X = train_np[:,1:]
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X,y)
    print(pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)}))
    return clf


def testPredict(df_test,clf):
    test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(test)
    result = pd.DataFrame( {'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
    result.to_csv("./predictions.csv", index=False)

import pandas as pd
data_train = pd.read_csv("train.csv")
data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
data_train = feature_factorization(data_train)
data_train = set_scale(data_train)
clf = train(data_train)

# print data_train

# data_test = pd.read_csv("./test.csv")
# data_test.loc[ (data_test.Fare.isnull()),'Fare' ] = 0
# tmp_df = data_test[ ['Age','Fare','Parch','SibSp','Pclass']]
# null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# X = null_age[:,1:]
# predictedAges = rfr.predict(X)
# data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges
# # data_test, rfr = set_missing_ages(data_test)
# data_test = set_Cabin_type(data_test)
# data_test = feature_factorization(data_test)
# data_test = set_scale(data_test)
# testPredict(data_test,clf)
# print(data_test)



