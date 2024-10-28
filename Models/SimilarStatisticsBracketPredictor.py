import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, accuracy_score, log_loss
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import time

cbb = pd.read_csv('RandomForestProject/cbb_more.csv')

del cbb['CONF'], cbb['GAMES'], cbb['W'], cbb['L']
cbb = cbb.drop(columns={'QUAD NO', 'TEAM ID', 'ROUND'})

cbb_test = cbb.loc[cbb['YEAR'] == 2024]
cbb_train = cbb.loc[cbb['YEAR'] != 2024]


matchups = pd.read_csv('RandomForestProject/Tournament Matchups.csv')
matchups.drop(columns={'BY YEAR NO', 'BY ROUND NO', 'SEED', 'ROUND', 'CURRENT ROUND'}, inplace=True)
# NEED TO DEAL WITH FIRST FOUR MATCHUPS
matchups = matchups.drop([2, 3, 8, 9, 56, 57, 60, 61]).reset_index(drop=True)
# Combine rows in pairs
combined_rows = []
for i in range(0, len(matchups), 2):
    row1 = matchups.iloc[i]
    row2 = matchups.iloc[i+1]
    
    combined_row = {
        'YEAR': row1['YEAR'],
        'TEAM NO': row1['TEAM NO'],
        'TEAM': row1['TEAM'],
        'SCORE': row1['SCORE'],
        'OPP TEAM NO': row2['TEAM NO'],
        'OPP TEAM': row2['TEAM'],
        'OPP SCORE': row2['SCORE']
    }
    combined_row_2 = {
        'YEAR': row1['YEAR'],
        'TEAM NO': row2['TEAM NO'],
        'TEAM': row2['TEAM'],
        'SCORE': row2['SCORE'],
        'OPP TEAM NO': row1['TEAM NO'],
        'OPP TEAM': row1['TEAM'],
        'OPP SCORE': row1['SCORE']
        }
    
    combined_rows.append(combined_row)
    combined_rows.append(combined_row_2)

# Create the new DataFrame
combined_matchups = pd.DataFrame(combined_rows)

matchups_test = combined_matchups.loc[combined_matchups['YEAR'] == 2024].reset_index(drop=True)
matchups_train = combined_matchups.loc[combined_matchups['YEAR'] != 2024].reset_index(drop=True)
matchups_train['WIN ID'] = np.nan
matchups_train['RESULT'] = np.nan

for index, row in matchups_train.iterrows():
    if row['SCORE'] > row['OPP SCORE']:
        matchups_train.at[index, 'WIN ID'] = matchups_train.at[index, 'TEAM NO']
        matchups_train.at[index, 'RESULT'] = 1
    elif row['SCORE'] < row['OPP SCORE']:
        matchups_train.at[index, 'WIN ID'] = matchups_train.at[index, 'OPP TEAM NO']
        matchups_train.at[index, 'RESULT'] = 0
matchups_train = matchups_train[['TEAM NO', 'OPP TEAM NO', 'RESULT']]

end = pd.merge(cbb_train, matchups_train, on='TEAM NO')
cbb_opponent = cbb_train.add_prefix('OPP_')
end = pd.merge(end, cbb_opponent, left_on='OPP TEAM NO', right_on='OPP_TEAM NO')
del end['OPP_TEAM NO'], end['OPP_TEAM'], end['OPP_YEAR']
end = end.filter(regex='^(?!.* RANK)', axis=1)

null_count = end.isnull().sum()
incomplete_cols = list(end.columns[null_count != 0])

first_columns_removed = {'K TEMPO', 'K OFF', 'K DEF', 'BADJ EM', 'BADJ O', 'BADJ D', 'RAW T', 
                  'BLKED%', 'BADJ T', 'OP FT%', 'QUAD ID', 'OP DREB%', 'OP OREB%', '2PTRD', 
                  'BLK%', 'OP AST%', '2PTR', '3PTRD', 'PPPO', 'PPPD', 'CONF ID', 'DREB%', 'WIN%'}
end.drop(columns=first_columns_removed, inplace=True)

opp_columns_removed = {'OPP_' + col for col in first_columns_removed}
end.drop(columns=opp_columns_removed, inplace=True)


# Creating the model
our = end.copy()

our.drop(columns={'YEAR', 'TEAM NO', 'TEAM', 'OPP TEAM NO'}, inplace=True)

# This is the target, what we want to predict
y_train = our['RESULT'].astype(int)

xgb_model = xgb.XGBClassifier()
logreg_model = LogisticRegression(max_iter=1000)

def custom_scorer(estimator, X, y):
    # Predict the classes and probabilities
    y_pred = estimator.predict(X)
    y_prob = estimator.predict_proba(X)
    
    # Calculate accuracy and log loss
    accuracy = accuracy_score(y, y_pred)
    logloss = log_loss(y, y_prob)
    
    # Scale the scores using min-max scaling
    scaled_acc = (accuracy - 0) / (1 - 0)  # accuracy is already in range 0-1
    min_logloss, max_logloss = 0, np.log(2)  # log loss for a binary classification ranges from 0 to log(2)
    scaled_logloss = (logloss - min_logloss) / (max_logloss - min_logloss)
    
    # Combine the scaled scores
    combined_score = 0.5 * scaled_acc + 0.5 * (1 - scaled_logloss)  # invert log loss for combination
    
    return combined_score

X_train = our.drop(columns={'RESULT'})
selected_columns = X_train.columns

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

n_features_to_test = 3
cv_folds_to_test = 14

sfs_test = SequentialFeatureSelector(estimator=xgb_model, n_features_to_select=n_features_to_test, 
                                     direction='forward', cv=TimeSeriesSplit(n_splits=cv_folds_to_test), 
                                     scoring=custom_scorer)

start_time = time.time()

# Fit the Sequential Feature Selector
sfs_test.fit(X_train, y_train)

end_time = time.time()
elapsed_time = end_time - start_time

estimated_full_time = elapsed_time * (13 / n_features_to_test) * (14 / cv_folds_to_test)
