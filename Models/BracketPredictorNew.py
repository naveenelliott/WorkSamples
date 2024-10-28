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
y_train = our['RESULT']

#xgb_model = xgb.XGBClassifier(eval_metric='logloss')
logreg_model = LogisticRegression(max_iter=1000)

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

# This is me getting an idea of which features are useful and which we should remove before doing more testing
sfs = SequentialFeatureSelector(logreg_model, n_features_to_select=13, direction='forward', cv=split, scoring=custom_scorer)

# Fitting the sequential feature selector
sfs.fit(X_train, y_train)

# transforming the training data to fit the sequential feature selector
X_train_selected = sfs.transform(X_train)

# fitting the model
logreg_model.fit(X_train_selected, y_train)

predictors = selected_columns[sfs.get_support()] 

matchups_test['WIN ID'] = np.nan
matchups_test['RESULT'] = np.nan

for index, row in matchups_test.iterrows():
    if row['SCORE'] > row['OPP SCORE']:
        matchups_test.at[index, 'WIN ID'] = matchups_test.at[index, 'TEAM NO']
        matchups_test.at[index, 'RESULT'] = 1
    elif row['SCORE'] < row['OPP SCORE']:
        matchups_test.at[index, 'WIN ID'] = matchups_test.at[index, 'OPP TEAM NO']
        matchups_test.at[index, 'RESULT'] = 0
matchups_test = matchups_test[['TEAM NO', 'OPP TEAM NO', 'RESULT']]

end_test = pd.merge(cbb_test, matchups_test, on='TEAM NO')
cbb_opponent = cbb_test.add_prefix('OPP_')
end_test = pd.merge(end_test, cbb_opponent, left_on='OPP TEAM NO', right_on='OPP_TEAM NO')

end_test_teams = end_test[['TEAM', 'SEED', 'OPP_TEAM', 'OPP_SEED']]

del end_test['OPP_TEAM NO'], end_test['OPP_TEAM'], end_test['OPP_YEAR']
end_test = end_test.filter(regex='^(?!.* RANK)', axis=1)
baseline_test = end_test.copy()

null_count = end_test.isnull().sum()
incomplete_cols = list(end_test.columns[null_count != 0])

first_columns_removed = {'K TEMPO', 'K OFF', 'K DEF', 'BADJ EM', 'BADJ O', 'BADJ D', 'RAW T', 
                  'BLKED%', 'BADJ T', 'OP FT%', 'QUAD ID', 'OP DREB%', 'OP OREB%', '2PTRD', 
                  'BLK%', 'OP AST%', '2PTR', '3PTRD', 'PPPO', 'PPPD', 'CONF ID', 'DREB%', 'WIN%', 
                  '3PT%D', 'FTRD', 'EXP', '3PT%', '2PT%', 'ELITE SOS', '3PTR', 'EFG%D', 'AVG HGT',
                  'KADJ T', '2PT%D', 'EFF HGT', 'FT%', 'TOV%', 'SEED', 'EFG%'}
end_test.drop(columns=first_columns_removed, inplace=True)

opp_columns_removed = {'OPP_' + col for col in first_columns_removed}
end_test.drop(columns=opp_columns_removed, inplace=True)


end_test.drop(columns={'YEAR', 'TEAM NO', 'TEAM', 'OPP TEAM NO'}, inplace=True)

# This is the target, what we want to predict
y_test = end_test['RESULT']

X_test = end_test.drop(columns={'RESULT'})
selected_columns = X_test.columns

scaler = MinMaxScaler()
X_test = scaler.fit_transform(X_test)
X_test_scaled = sfs.transform(X_test)

y_pred = logreg_model.predict(X_test_scaled)

end_test['PRED_RESULT'] = y_pred

testing_accuracy = accuracy_score(end_test['RESULT'], end_test['PRED_RESULT'])


baseline_test = baseline_test[['YEAR', 'SEED', 'OPP_SEED', 'WIN%', 'OPP_WIN%', 'RESULT']]

baseline_test['BASELINE_RESULT'] = baseline_test['SEED'] < baseline_test['OPP_SEED']

baseline_test.loc[baseline_test['SEED'] == baseline_test['OPP_SEED'], 'BASELINE_RESULT'] = (baseline_test['WIN%'] > baseline_test['OPP_WIN%'])

baseline_test['BASELINE_RESULT'] = baseline_test['BASELINE_RESULT']

testing_baseline_accuracy = accuracy_score(baseline_test['RESULT'], baseline_test['BASELINE_RESULT'])


for index, row in matchups_info.iterrows():
    if row['SCORE'] > row['OPP SCORE']:
        matchups_info.at[index, 'WIN ID'] = matchups_info.at[index, 'TEAM NO']
        matchups_info.at[index, 'RESULT'] = 1
    elif row['SCORE'] < row['OPP SCORE']:
        matchups_info.at[index, 'WIN ID'] = matchups_info.at[index, 'OPP TEAM NO']
        matchups_info.at[index, 'RESULT'] = 0
matchups_info = matchups_info[['TEAM', 'OPP TEAM']]

y_prob = logreg_model.predict_proba(X_test_scaled)

# Extract the confidence level for the predicted class
end_test['CONFIDENCE'] = y_prob.max(axis=1)

# Combine the predictions with the actual results and confidence levels
end_test = end_test[['RESULT', 'PRED_RESULT', 'CONFIDENCE']]

# Merge with team information for better interpretability
end_test = end_test.merge(end_test_teams, left_index=True, right_index=True)

end_test['matchup_id'] = end_test.apply(lambda row: tuple(sorted([row['TEAM'], row['OPP_TEAM']])), axis=1)

# Sort the DataFrame by 'matchup_id' and 'CONFIDENCE' in descending order
end_test = end_test.sort_values(by=['matchup_id', 'CONFIDENCE'], ascending=[True, False])

# Drop duplicates within the same matchup, keeping the row with the highest confidence
end_test = end_test.drop_duplicates(subset='matchup_id', keep='first')

# Drop the temporary 'matchup_id' column
end_test = end_test.drop(columns='matchup_id')

end_test.reset_index(drop=True, inplace=True)

test_data_accuracy = accuracy_score(end_test['RESULT'], end_test['PRED_RESULT'])
