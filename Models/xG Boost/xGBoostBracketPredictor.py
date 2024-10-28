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
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression

cbb = pd.read_csv('RandomForestProject/cbb_more.csv')

del cbb['CONF'], cbb['GAMES'], cbb['W'], cbb['L']
cbb = cbb.drop(columns={'QUAD NO', 'TEAM ID', 'ROUND'})
cbb = cbb.loc[cbb['YEAR'] != 2024]

cbb_test = cbb.loc[cbb['YEAR'] == 2023]
cbb_train = cbb.loc[cbb['YEAR'] != 2023]


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

combined_matchups = combined_matchups.loc[combined_matchups['YEAR'] != 2024].reset_index(drop=True)
matchups_test = combined_matchups.loc[combined_matchups['YEAR'] == 2023].reset_index(drop=True)
matchups_info = combined_matchups.loc[combined_matchups['YEAR'] == 2023].reset_index(drop=True)
matchups_train = combined_matchups.loc[combined_matchups['YEAR'] != 2023].reset_index(drop=True)
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
baseline = end.copy()

null_count = end.isnull().sum()
incomplete_cols = list(end.columns[null_count != 0])

first_columns_removed = {'K TEMPO', 'K OFF', 'K DEF', 'BADJ EM', 'BADJ O', 'BADJ D', 'RAW T', 
                  'BLKED%', 'BADJ T', 'OP FT%', 'QUAD ID', 'OP DREB%', 'OP OREB%', '2PTRD', 
                  'BLK%', 'OP AST%', '2PTR', '3PTRD', 'PPPO', 'PPPD', 'CONF ID', 'DREB%', 'WIN%', 
                  '3PT%D', 'FTRD', 'EXP', '3PT%', '2PT%', 'ELITE SOS', '3PTR', 'EFG%D', 'AVG HGT',
                  'KADJ T', '2PT%D', 'EFF HGT', 'FT%', 'TOV%', 'SEED', 'EFG%'}
end.drop(columns=first_columns_removed, inplace=True)

opp_columns_removed = {'OPP_' + col for col in first_columns_removed}
end.drop(columns=opp_columns_removed, inplace=True)


# Creating the model
our = end.copy()

our.drop(columns={'YEAR', 'TEAM NO', 'TEAM', 'OPP TEAM NO'}, inplace=True)

# This is the target, what we want to predict
y_train = our['RESULT'].astype(int)


xgb_model = xgb.XGBClassifier(eval_metric='logloss')


# this scoring system combines accuracy and logloss, two of the most important metrics in feature selection for model building
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

# Creating a time series split and splitting into 15 sets of training and test data
# This is used for cross validation
split = TimeSeriesSplit(n_splits=14)

# List to hold cross-validation scores
cv_scores = []

n_features_to_select = range(1, X_train.shape[1])

# Iterate over the number of features to select
for n in n_features_to_select:
    # This runs through and selects the optimal amount of features
    sfs = SequentialFeatureSelector(estimator=xgb_model, 
                                    n_features_to_select=n, 
                                    direction='forward', 
                                    scoring=custom_scorer, 
                                    cv=split,
                                    n_jobs=-1)
    
    # Fitting the data with the sequential feature selector
    sfs.fit(X_train, y_train)
    # Cross-validate the model with the selected features
    score = np.mean(cross_val_score(xgb_model, sfs.transform(X_train), y_train, cv=split, scoring='accuracy', n_jobs=-1))
    cv_scores.append(score)

# Finding the optimal number of features
optimal_n = n_features_to_select[np.argmax(cv_scores)]
print("Optimal number of features:", optimal_n)
print("Best cross-validation accuracy:", max(cv_scores))

# This is plotting the cross validation and mean squared error so we can visualize the scores and 
# amount of features
plt.figure(figsize=(10, 6))
plt.plot(n_features_to_select, cv_scores, marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation Log Loss and Accuracy')
plt.title('Cross-Validation Log Loss and Accuracy vs. Number of Features')
plt.grid()
plt.xticks(n_features_to_select)
plt.show()

# Either 9,12 or 14 would be the best
# try 14 first

# This is me getting an idea of which features are useful and which we should remove before doing more testing
sfs = SequentialFeatureSelector(xgb_model, n_features_to_select=14, direction='forward', cv=split, scoring=custom_scorer)

# Fitting the sequential feature selector
sfs.fit(X_train, y_train)

# transforming the training data to fit the sequential feature selector
X_train_selected = sfs.transform(X_train)

# fitting the model
xgb_model.fit(X_train_selected, y_train)

predictors = selected_columns[sfs.get_support()] 

cv_accuracy_scores = cross_val_score(xgb_model, X_train_selected, y_train, cv=split, scoring='accuracy', n_jobs=-1)
average_accuracy = np.mean(cv_accuracy_scores)
print(f'Model Accuracy {average_accuracy}')

# I don't know if this is right for sure
log_losses = []

for train_index, val_index in split.split(X_train_selected):
    X_train_fold, X_val_fold = X_train_selected[train_index], X_train_selected[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Fit the model on the training fold
    xgb_model.fit(X_train_fold, y_train_fold)
    
    # Predict probabilities on the validation fold
    y_pred_proba = xgb_model.predict_proba(X_val_fold)
    
    # Calculate log loss for the validation fold
    fold_log_loss = log_loss(y_val_fold, y_pred_proba)
    log_losses.append(fold_log_loss)

# Calculate the average log loss
average_log_loss = np.mean(log_losses)

print("Average Cross-Validation Log Loss:", average_log_loss)