from locale import normalize

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import re
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

pd.set_option('display.width', 256)


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
all_data = pd.concat([train_data,test_data])

print("===== survived by class and sex")
print(train_data.groupby(['Pclass' , 'Sex'])['Survived'].value_counts(normalize=True))

describe_fields = ['Age' , 'Fare' , 'Pclass' , 'SibSp' , 'Parch']

print('-----Train: males')
print(train_data[train_data['Sex'] == 'male'][describe_fields].describe())

print('-----Test: males')
print(test_data[test_data['Sex'] == 'male'][describe_fields].describe())

print('-----Train: females')
print(train_data[train_data['Sex'] == 'female'][describe_fields].describe())

print('-----Test: females')
print(test_data[test_data['Sex'] == 'female'][describe_fields].describe())

class DataDigest:

    def __init__(self):
        self.ages = None
        self.fares = None
        self.titles = None
        self.cabins = None
        self.families = None
        self.tickets = None

def get_title(name):
    if pd.isnull(name):
        return 'Null'

    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1).lower()
    else:
        return 'None'

def get_family(row):
    last_name = row['Name'].split(',')[0]
    if last_name:
        family_size = 1 + row['Parch'] + row['SibSp']
        if family_size > 3:
            return f'{last_name.lower()}_{family_size}'
        else:
            return 'nofamily'
    else:
        return 'unknown'

data_digest = DataDigest()
data_digest.ages = all_data.groupby('Sex')['Age'].median()
data_digest.fares = all_data.groupby('Pclass')['Fare'].median()
data_digest.titles = pd.Index(test_data.apply(get_family, axis=1).unique())
data_digest.families = pd.Index(test_data.apply(get_family, axis=1).unique())
data_digest.cabins = pd.Index(test_data["Cabin"].fillna('unknown').unique())
data_digest.tickets = pd.Index(test_data['Ticket'].fillna('unknown').unique())

def get_index(item, index):
    if pd.isnull(item):
        return -1

    try:
        return index.get_loc(item)
    except KeyError:
        return -1

def munge_data (data, digest):
    data['AgeF'] = data.apply(lambda r: digest.ages[r['Sex']] if pd.isnull(r['Age']) else r['Age'], axis =1)
    data['FareF'] = data.apply(lambda r:digest.fares[r['Pclass']] if pd.isnull(r['Fare']) else r['Fare'], axis = 1)

    genders = {'male': 1, 'female': 0}
    data['SexF'] = data['Sex'].apply(lambda s: genders.get(s))

    gender_dummies = pd.get_dummies(data['Sex'], prefix='SexD' , dummy_na=False)
    data = pd.concat([data, gender_dummies], axis=1)

    embarkerments = {'U': 0, 'S': 1, 'C': 2, 'Q': 3}
    data['EmbarkedF'] = data['Embarked'].fillna('U').apply(lambda e: embarkerments.get(e))

    embarkerment_dummies = pd.get_dummies(data['Embarked'], prefix='EmbarkedD' , dummy_na=False)
    data = pd.concat([data, embarkerment_dummies], axis=1)

    data['RelativesF'] = data['Parch'] + data['SibSp']

    data['SingleF'] = data['RelativesF'].apply(lambda  r: 1 if r == 0 else 0)

    decks = {"U": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
    data['DeckF'] = data['Cabin'].fillna('U').apply(lambda c: decks.get(c[0]), -1)

    deck_dummies = pd.get_dummies(data['Cabin'].fillna('U').apply(lambda c: c[0]), prefix='DeckD' , dummy_na=False)
    data = pd.concat([data, deck_dummies], axis=1)

    title_dummies = pd.get_dummies(data['Name'].apply(lambda n: get_title(n)), prefix='TitleD' , dummy_na=False)
    data = pd.concat([data, title_dummies], axis=1)

    data['CabinF'] = data['Cabin'].fillna('unknown').apply(lambda c:get_index(c, digest.cabins))

    data['TitleF'] = data['Name'].apply(lambda n: get_index(get_title(n), digest.titles))

    data['TicketF'] = data['Ticket'].apply(lambda t: get_index(t, digest.tickets))

    data['FamilyF'] = data.apply(lambda r: get_index(get_family(r), digest.families), axis=1)

    age_bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
    data['AgeR'] = pd.cut(data['Age'].fillna(-1), bins=age_bins).astype(object)

    return data


train_data_munged = munge_data(train_data, data_digest)
test_data_munged = munge_data(test_data, data_digest)
all_data_munged = pd.concat([test_data_munged,test_data_munged])

#all_data_munged.to_csv('middle_result.csv')

predictors = ["Pclass",
              "AgeF",
              "TitleF",
              "TitleD_mr", "TitleD_mrs", "TitleD_miss", "TitleD_master", "TitleD_ms",
              "TitleD_col", "TitleD_rev", "TitleD_dr",
              "CabinF",
              "DeckF",
              "DeckD_U", "DeckD_A", "DeckD_B", "DeckD_C", "DeckD_D", "DeckD_E", "DeckD_F", "DeckD_G",
              "FamilyF",
              "TicketF",
              "SexF",
              "SexD_male", "SexD_female",
              "EmbarkedF",
              "EmbarkedD_S", "EmbarkedD_C", "EmbarkedD_Q",
              "FareF",
              "SibSp", "Parch",
              "RelativesF",
              "SingleF"]

scaler = StandardScaler()                         # приведение к одному диапазону
scaler.fit(all_data_munged[predictors])

train_data_scaled = scaler.transform(train_data_munged[predictors])
test_data_scaled = scaler.transform(test_data_munged[predictors])

print('----------survived by age')
print(train_data_munged.groupby(['AgeR'])['Survived'].value_counts(normalize=True))

print('--------survived by gender and age')
print(train_data_munged.groupby(['Sex' ,'AgeR'])['Survived'].value_counts(normalize=True))

print('---------survived by class and age')
print(train_data_munged.groupby(['Pclass' , 'AgeR'])['Survived'].value_counts(normalize = True))

#sns.pairplot(train_data_munged, vars = ['AgeF' , 'Pclass' , 'SexF'] , hue = 'Survived' , dropna=True)  # отображение зависимости выживания от классов
#plt.show()

#selector = SelectKBest(f_classif, k=5)                                           # важность показателей
#selector.fit(train_data_munged[predictors], train_data_munged['Survived'])

#scores = -np.log10(selector.pvalues_)

#plt.bar(range(len(predictors)), scores)
#plt.xticks(range(len(predictors)), predictors, rotation='vertical')
#plt.show()

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

alg_ngbh = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(alg_ngbh, train_data_scaled, train_data_munged['Survived'], cv=cv, n_jobs=1)
print(f'Accuracy (k-neighbors):{scores.mean()}/{scores.std()}')

alg_sgd = SGDClassifier(random_state=1)
scores = cross_val_score(alg_sgd, train_data_scaled, train_data_munged['Survived'], cv=cv, n_jobs=1)
print("Accuracy (sgd): {}/{}".format(scores.mean(), scores.std()))

alg_svm = SVC(C=1.0)
scores = cross_val_score(alg_svm, train_data_scaled, train_data_munged['Survived'], cv=cv, n_jobs=1)
print("Accuracy (svm): {}/{}".format(scores.mean(), scores.std()))

alg_nbs = GaussianNB()
scores = cross_val_score(alg_nbs, train_data_scaled, train_data_munged['Survived'], cv=cv, n_jobs=1)
print("Accuracy (naive bayes): {}/{}".format(scores.mean(), scores.std()))

def linear_scorer(estimator, x, y):
    scorer_predictors = estimator.predict(x)

    scorer_predictors[scorer_predictors > 0.5] = 1
    scorer_predictors[scorer_predictors <= 0.5] = 0

    return metrics.accuracy_score(y, scorer_predictors)

alg_lnr = LinearRegression()
scores = cross_val_score(alg_lnr, train_data_scaled, train_data_munged['Survived'], cv=cv, n_jobs=1, scoring=linear_scorer)
print("Accuracy (linear regression): {}/{}".format(scores.mean(), scores.std()))

alg_log = LogisticRegression(random_state=1)
scores = cross_val_score(alg_log, train_data_scaled, train_data_munged['Survived'], cv=cv, n_jobs=1, scoring=linear_scorer)
print("Accuracy (logistic regression): {}/{}".format(scores.mean(), scores.std()))

alg_frst = RandomForestClassifier(random_state=1, n_estimators=500, min_samples_split=8, min_samples_leaf=2)
scores = cross_val_score(alg_frst, train_data_scaled, train_data_munged['Survived'], cv=cv, n_jobs=1)
print("Accuracy (random forest): {}/{}".format(scores.mean(), scores.std()))

alg_frst_model = RandomForestClassifier(random_state=1)
alg_frst_params = [{
    'n_estimators': [350,400,450],
    'min_samples_split': [6,8,10],
    'min_samples_leaf': [1,2,4]
}]

alg_frst_grid = GridSearchCV(alg_frst_model, alg_frst_params, cv=cv, refit=True, verbose=1, n_jobs=1)   # поиск лучших параметров для модели
alg_frst_grid.fit(train_data_scaled, train_data_munged['Survived'])
alg_frst_best = alg_frst_grid.best_estimator_
print("Accuracy (random forest auto): {} with params {}".format(alg_frst_grid.best_score_, alg_frst_grid.best_params_))

alg_xgb_model = xgb.XGBClassifier(use_label_encoder=False)
alg_xgb_params = [{
    'n_estimators':[230,250,270],
    'max_depth': [1,2,4],
    'learning_rate': [0.01, 0.02, 0.05]
}]

alg_xgb_grid = GridSearchCV(alg_xgb_model, alg_xgb_params, cv=cv, refit=True, verbose=1, n_jobs=1)
alg_xgb_grid.fit(train_data_scaled,train_data_munged['Survived'])
alg_xgb_best = alg_xgb_grid.best_estimator_
print("Accuracy (xgboost auto): {} with params {}".format(alg_xgb_grid.best_score_, alg_xgb_grid.best_params_))

alg_test = alg_frst_best
alg_test.fit(train_data_scaled, train_data_munged['Survived'])

predictions = alg_test.predict(test_data_scaled)

submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})

submission.to_csv('titanic_submission.csv' , index=False)

alg_test1 = alg_xgb_best
alg_test1.fit(train_data_scaled, train_data_munged['Survived'])

predictions1 = alg_test1.predict(test_data_scaled)

submission1 = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions1
})

submission1.to_csv('titanic_submission1.csv' , index=False)

