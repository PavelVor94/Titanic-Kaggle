import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import pylab as pl

data = pd.read_csv('train.csv')

#data.pivot_table('PassengerId' , 'Pclass' , 'Survived' , 'count' ).plot(kind='bar' , stacked=True)
#plt.show()

#fig, axes = plt.subplots(ncols=2)
#data.pivot_table('PassengerId' , ['SibSp'], 'Survived' , 'count').plot(ax = axes[0], title= 'SibSp')
#data.pivot_table('PassengerId' , ['Parch'], 'Survived' , 'count').plot(ax = axes[1], title='Parch')
#plt.show()

print('количество заполненых номеров кают: ' ,data.PassengerId[data.Cabin.notnull()].count())
print('количество заполненых возрастов: ' ,data.PassengerId[data.Age.notnull()].count())

data.Age[data.Age.isnull()] = data.Age.mean()

print('количество незаполненых портов посадки: ' ,data.PassengerId[data.Embarked.isnull()].count())

MaxPassEmbarked = data.groupby('Embarked').count()['PassengerId']
data.Embarked[data.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]

print('количество незаполненых цен билетов: ' ,data.PassengerId[data.Fare.isnull()].count())

data = data.drop(['PassengerId' , 'Name' , 'Ticket' , 'Cabin'], axis=1)
print(data.head())

label = LabelEncoder()
dicts = {}

label.fit(data.Sex.drop_duplicates())
dicts['Sex'] = list(label.classes_)
data.Sex = label.transform(data.Sex)

label.fit(data.Embarked.drop_duplicates())
dicts['Embarked'] = list(label.classes_)
data.Embarked = label.transform(data.Embarked)

print(data.head())

test = pd.read_csv('test.csv')
test.Age[test.Age.isnull()] = test.Age.mean()
test.Fare[test.Fare.isnull()] = test.Fare.median()
MaxPassEmbarked = test.groupby('Embarked').count()['PassengerId']
test.Embarked[test.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
result = pd.DataFrame(test.PassengerId)
test = test.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)

label.fit(dicts['Sex'])
test.Sex = label.transform(test.Sex)

label.fit(dicts['Embarked'])
test.Embarked = label.transform(test.Embarked)

target = data.Survived
train = data.drop(['Survived'], axis=1)

kfold = 5

itog_val = {}

print(train.head())

ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = train_test_split(train, target, test_size= 0.25)

model_rfc = RandomForestClassifier(n_estimators=80, max_features='auto' , criterion='entropy' , max_depth=4)
model_knc = KNeighborsClassifier(n_neighbors=18)
model_lr = LogisticRegression(penalty='l2' , tol=0.01)
model_svc = svm.SVC()

scores = cross_val_score(model_rfc, train, target, cv=kfold)
itog_val['RandomForestClassifier'] = scores.mean()
scores = cross_val_score(model_knc, train, target, cv = kfold)
itog_val['KNeighborsClassifier'] = scores.mean()
scores = cross_val_score(model_lr, train, target, cv = kfold)
itog_val['LogisticRegression'] = scores.mean()
scores = cross_val_score(model_svc, train, target, cv = kfold)
itog_val['SVC'] = scores.mean()

#pd.DataFrame.from_dict(data=itog_val, orient='index').plot(kind='bar' , legend = False)
#plt.show()

pl.clf()
plt.figure(figsize=(8,6))

model_svc.probability = True
probas = model_svc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('SVC', roc_auc))

probas = model_rfc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('RandonForest',roc_auc))

#KNeighborsClassifier
probas = model_knc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('KNeighborsClassifier',roc_auc))

#LogisticRegression
probas = model_lr.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
roc_auc  = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LogisticRegression',roc_auc))
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.legend(loc=0, fontsize='small')
#pl.show()

model_rfc.fit(train, target)

result.insert(1,'Survived' , model_rfc.predict(test))
result.to_csv('result.csv' , index=False)
