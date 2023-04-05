from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedShuffleSplit, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from gtzantest import Preprocessing
import pandas as pd

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

X, y = Preprocessing.load_data()

split = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=42)
for train_indices, test_indices in split.split(X, y):
    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LinearSVC())
])
param_grid = [
    {'scaler': [MaxAbsScaler(), MinMaxScaler(), StandardScaler()],
     'classifier': [AdaBoostClassifier(), ExtraTreesClassifier(), GradientBoostingClassifier()],
     'classifier__n_estimators': [10, 100, 1000]},
    {'scaler': [MaxAbsScaler(), MinMaxScaler(), StandardScaler()],
     'classifier': [RandomForestClassifier()],
     'classifier__n_estimators': [10, 100, 1000],
     'classifier__max_features': [5, 10, 100],
     'classifier__criterion': ['gini', 'entropy'],
     'classifier__max_features': ['sqrt', 'log2', None]},
    {'scaler': [MaxAbsScaler(), MinMaxScaler(), StandardScaler()],
     'classifier': [SGDClassifier()],
     'classifier__max_iter': [100, 1000]},
    {'scaler': [MaxAbsScaler(), MinMaxScaler(), StandardScaler()],
     'classifier': [LinearSVC()],
     'classifier__C': [0.01, 0.5, 1.0]},
    {'scaler': [MaxAbsScaler(), MinMaxScaler(), StandardScaler()],
     'classifier': [SVC(gamma='auto')],
     'classifier__C': [0.01, 0.5, 1.0],
     'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
     'classifier__degree': [1, 3, 30]}
]
param_dist = {
    'scaler': [MaxAbsScaler(), MinMaxScaler(), StandardScaler()],
    'classifier': [AdaBoostClassifier(), ExtraTreesClassifier(), GradientBoostingClassifier(),
                   RandomForestClassifier(n_estimators=100), SGDClassifier(), LinearSVC(), SVC(gamma='auto')]
}

model_search = GridSearchCV(pipeline, param_grid, cv=3, iid=False, return_train_score=True)
# model_search = RandomizedSearchCV(pipeline, param_dist, n_iter=100, cv=3)
model = model_search.fit(X_train, pd.Series.ravel(y_train))