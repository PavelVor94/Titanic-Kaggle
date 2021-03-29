import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np



trd = pd.read_csv('train.csv')
tsd = pd.read_csv('test.csv')
td = pd.concat([trd,tsd], ignore_index=True, sort=False)
print(td.columns)

print(td.isnull().sum()) #пропущенные данные
#sns.heatmap(td.isnull(), cbar = False).set_title('Missing')
#plt.show()
print(td.nunique()) # уникальные значения

td['Family'] = td.Parch + td.SibSp
td["Is_Alone"] = td.Family == 0
td['Fare_Category'] = pd.cut(td['Fare'], bins=[0,7.9,14.45,31.28,120], labels=['Low' , 'Mid' , 'Hid_Mid' , 'High'])

td.Embarked.fillna(td.Embarked.mode()[0], inplace=True)
td.Cabin = td.Cabin.fillna('NA')

td['Salutation'] = td.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

grp = td.groupby(['Sex' , 'Pclass'])

grp.Age.apply(lambda x: x.fillna(x.median()))

td.Age.fillna(td.Age.median, inplace=True)

td['Sex'] = LabelEncoder().fit_transform(td['Sex'])

pd.get_dummies(td.Embarked, prefix='Emb' , drop_first=True)
td.drop(['Pclass', 'Fare','Cabin', 'Fare_Category','Name','Salutation', 'Ticket','Embarked', 'SibSp', 'Parch', 'Age'], axis=1, inplace=True)

# Data to be predicted
X_to_be_predicted = td[td.Survived.isnull()]
X_to_be_predicted = X_to_be_predicted.drop(['Survived'], axis = 1)
# X_to_be_predicted[X_to_be_predicted.Age.isnull()]
# X_to_be_predicted.dropna(inplace = True) # 417 x 27
#Training data
train_data = td
train_data = train_data.dropna()
feature_train = train_data['Survived']
label_train = train_data.drop(['Survived'], axis = 1)
##Gaussian
clf = GaussianNB()
x_train, x_test, y_train, y_test = train_test_split(label_train, feature_train, test_size=0.2)
clf.fit(x_train, np.ravel(y_train))
print("NB Accuracy: "+repr(round(clf.score(x_test, y_test) * 100, 2)) + "%")
result_rf=cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')
print('The cross validated score for Random forest is:',round(result_rf.mean()*100,2))
y_pred = cross_val_predict(clf,x_train,y_train,cv=10)
#sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="summer")
#plt.title('Confusion_matrix for NB', y=1.05, size=15)
#plt.show()

##Random forest
clfw = RandomForestClassifier(criterion='entropy',
    n_estimators=700,
    min_samples_split=10,
    min_samples_leaf=1,
    max_features='auto',
    oob_score=True,
    random_state=1,
    n_jobs=-1)
x_train, x_test, y_train, y_test = train_test_split(label_train, feature_train, test_size=0.2)
clfw.fit(x_train, np.ravel(y_train))
print("RF Accuracy: "+str((round(clf.score(x_test, y_test) * 100, 2))) + "%")
print(1)
result_rf=cross_val_score(clfw,x_train,y_train,cv=10,scoring='accuracy')

print('The cross validated score for Random forest is:',round(result_rf.mean()*100,2))
y_pred = cross_val_predict(clfw,x_train,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix for RF', y=1.05, size=15)
plt.show()

result = clfw.predict(X_to_be_predicted)
submission = pd.DataFrame({'PassengerId':X_to_be_predicted.PassengerId,'Survived':result})
submission.Survived = submission.Survived.astype(int)
print(submission.shape)
filename = 'Titanic Predictions.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)