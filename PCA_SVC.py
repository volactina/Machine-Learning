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
from gtzantest import spark_Preprocessing
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

X, y = spark_Preprocessing.load_data()


split = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=42)
for train_indices, test_indices in split.split(X, y):
    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]

pca=PCA(n_components=10)
pca.fit(X_train)
# print(pca.explained_variance_ratio_)
X_train=pca.transform(X_train)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LinearSVC())
])
param_grid = [
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

model_search = GridSearchCV(pipeline, param_grid, cv=3, iid=False, return_train_score=True,scoring="accuracy",n_jobs=4)
# model_search = RandomizedSearchCV(pipeline, param_dist, n_iter=100, cv=3,scoring="accuracy")
model = model_search.fit(X_train, pd.Series.ravel(y_train))
print(model.best_score_)
# print(model.best_estimator_.feature_importances_)
# print(model.best_params_)
# print("parameters")
# print(model.cv_results_)
pickle.dump(model,open("/usr/local/hadoop-2.7.7/bin/model_pca_svc",'wb'))
GENRES_LIST = ["blues",
                   "classical",
                   "country",
                   "disco",
                   "hiphop",
                   "jazz",
                   "metal",
                   "pop",
                   "reggae",
                   "rock"]
X_test=pca.transform(X_test)
y_pred=model.predict(X_test)
print("report:")
print(classification_report(y_test,y_pred,target_names=GENRES_LIST))
confusion_matrix=confusion_matrix(y_test,y_pred)
confusion_matrix=confusion_matrix.astype("float")/confusion_matrix.sum(axis=1)[:,np.newaxis]
print(confusion_matrix.diagonal())
print(accuracy_score(y_test,y_pred))