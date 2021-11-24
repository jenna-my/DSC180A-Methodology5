import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
import joblib

# Build Decision Tree Regressor Model
def model_build(features, tag, predictions_fp, mdl_fp):
    X_train, X_test, y_train, y_test = train_test_split(features, tag, test_size=0.2)
    dtr = DecisionTreeRegressor(max_depth = 2)
    dtr.fit(X_train, y_train)
    predictions = dtr.predict(X_test)
    
    out = pd.concat([pd.Series(predictions), y_test], axis=1)
    out.to_csv(predictions_fp)
    joblib.dump(dtr, mdl_fp, compress=1)
    
    return out
    

# def models():
    
#     mdls = {
#         "LogisticRegression": LogisticRegression,
#         "RandomForestClassifier": RandomForestClassifier
#     }

#     return mdls


# def model_build(
#         features,
#         tag,
#         predictions_fp,
#         mdl_fp,
#         modeltype, 
#         test_size, 
#         **params):

#     X_train, X_test, y_train, y_test = train_test_split(
#         features, tag, test_size=test_size)

#     mdl = models()[modeltype] # get model from dict
#     mdl = mdl(**params) # instantiate model w/given params

#     pl = Pipeline([
#         ('one-hot', OneHotEncoder(handle_unknown='ignore')),
#         ('mdl', mdl)
#     ])

#     pl.fit(X_train, y_train)
#     predictions = pl.predict(X_test)
#     out = pd.concat([pd.Series(predictions), y_test], axis=1)

#     out.to_csv(predictions_fp)
#     joblib.dump(pl, mdl_fp, compress=1)
    
#     return out