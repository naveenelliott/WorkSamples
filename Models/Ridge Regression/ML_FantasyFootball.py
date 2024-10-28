import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# loading in dataset
ffdata = pd.read_csv('RandomForestProject/ffdata.csv')

# Filtering out players with just one row
ffdata = ffdata.groupby('PlayerID', group_keys=False).filter(lambda x: x.shape[0] > 1)

# Deciding which position to build a model for
position_counts = ffdata['FantPos'].value_counts()
ffdata = ffdata.loc[ffdata['FantPos'] == 'RB']
temp = ffdata.drop_duplicates(subset=['PlayerID'])
rb_counts = temp['FantPos'].value_counts()

# Deleting team column because not optimal for machine learning (formatting issues)
del ffdata['Tm']

# Adding in another variable which is yards per carry
ffdata['YPC'] = ffdata['RushYds']/ffdata['RushAtt']

# Getting the PPR points for the next season, this will be the target
def next_season(player):
    player = player.sort_values('Year')
    player['Next_PPR'] = player['PPR'].shift(-1)
    return player

ffdata = ffdata.groupby('PlayerID', group_keys=False).apply(next_season)

# Removing players who have played less than 10 games total
# Plotting the histogram to see what the shape of the data looks like
outliers = ffdata.groupby('PlayerID')['G'].sum()
plt.hist(outliers, bins=10, edgecolor='black')
plt.title('Histogram of Total Games per Player')
plt.xlabel('Total Games')
plt.ylabel('Frequency')
plt.show()
outliers = outliers.loc[outliers <= 10]
ffdata = ffdata[~ffdata['PlayerID'].isin(outliers.index)]

# We can't have nulls in our dataset for ML
ffdata['YPC'] = ffdata['YPC'].fillna(0)
null_count = ffdata.isnull().sum()
incomplete_cols = list(ffdata.columns[null_count != 0])
# Removing the target column
incomplete_cols.remove('Next_PPR')
ffdata.drop(columns=incomplete_cols, inplace=True)

# Also dropping columns that deal with QB fantasy points b/c we are evaluating RBs
ffdata.drop(columns=['Cmp', 'Att', 'Yds', 'TD', 'Int'], inplace=True)


# Getting training and testing data
train_data = ffdata.loc[ffdata['Year'] != 2021]
# this data we will use to predict the Next_PPR for the 2021 season or the 2022 season PPR
test_data = ffdata.loc[ffdata['Year'] == 2021]
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

test_sum = len(test_data)
total_sum = len(ffdata)
test_percent = test_sum/total_sum
# Should be close to around 20% of the dataset and in this case it is ~15%


# Setting up ridge regression
rr = Ridge(alpha=1)
# Removing variables that aren't fit for ML and na rows
X_train = train_data.drop(columns=['Next_PPR', 'PlayerID', 'Year', 'FantPos', 'Player'])
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
removed_columns = ['Next_PPR', 'PlayerID', 'Year', 'FantPos', 'Player']
selected_columns = train_data.columns[~train_data.columns.isin(removed_columns)]

# When working with ridge regression, we need to scale the data
# MinMaxScaler scales the values to be between 0 and 1
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# This is the target, what we want to predict
y_train = train_data['Next_PPR']

# Creating a time series split and splitting into 4 sets of training and test data
# This is used for cross validation
split = TimeSeriesSplit(n_splits=4)

# Here, we will attempt to find the best amount of features that we should use based on the mean squared error
# and cross validation
n_features_to_select = range(1, X_train.shape[1])

# List to hold cross-validation scores
cv_scores = []

# Iterate over the number of features to select
for n in n_features_to_select:
    # This runs through and selects the optimal amount of features
    sfs = SequentialFeatureSelector(estimator=rr, 
                                    n_features_to_select=n, 
                                    direction='forward', 
                                    scoring='neg_mean_squared_error', 
                                    cv=split)
    # Fitting the data with the sequential feature selector
    sfs.fit(X_train, y_train)
    # Cross-validate the model with the selected features
    score = np.mean(cross_val_score(rr, sfs.transform(X_train), y_train, cv=split, scoring='neg_mean_squared_error'))
    cv_scores.append(score)

# Finding the optimal number of features
optimal_n = n_features_to_select[np.argmax(cv_scores)]
print("Optimal number of features:", optimal_n)
print("Best cross-validation neg MSE:", max(cv_scores))

# This is plotting the cross validation and mean squared error so we can visualize the scores and 
# amount of features
plt.figure(figsize=(10, 6))
plt.plot(n_features_to_select, cv_scores, marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation Negative MSE')
plt.title('Cross-Validation Negative MSE vs. Number of Features')
plt.grid()
plt.xticks(n_features_to_select)
plt.show()


# This creates the feature selector that selects the top 10 features from the data
sfs = SequentialFeatureSelector(rr, n_features_to_select=10, direction='forward', cv=split)

# Fitting the sequential feature selector
sfs.fit(X_train, y_train)

# transforming the training data to fit the sequential feature selector
X_train_selected = sfs.transform(X_train)

# fitting the model
rr.fit(X_train_selected, y_train)

# Getting the top 10 predictors
predictors = selected_columns[sfs.get_support()]  

# The top predictors are Age, Games, Rushing yds, Rushing TDs, Targets, receptions, receiving yards
# PPR for that season, Position Rank, and yards per carry

# Going through the same process as we did with the training set
X_test = test_data.drop(columns=['PlayerID', 'Year', 'FantPos', 'Player', 'Next_PPR'])
X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna() 
X_test_scaled = scaler.transform(X_test)
X_test_scaled = sfs.transform(X_test_scaled)

# Getting the predicted PPR values for the 2022 season
y_pred = rr.predict(X_test_scaled)
test_data['Predicted_PPR'] = y_pred

# Evaluating how accurate our predictions were
mse = mean_squared_error(test_data['Next_PPR'], test_data['Predicted_PPR'])

# How do we know if this is any good?
# This gives us the standard deviation of 84.58
test_data['PPR'].describe()

# taking the square root
# this is lower than the standard deviation, so this model is ok, but can be better
mse_sr = mse ** 0.5



# Getting the difference between actual values and their predictions
test_data['diff'] = abs(test_data['PPR'] - test_data['Predicted_PPR'])

test_data_limit = test_data.copy()

# STOPPED HERE
# Adding the rolling averages
# could be 4 idk need to run again
def add_rolling_averages(df, window=5):
    # Sorting by playerID and Year so that earlier years will go first
    df = df.sort_values(by=['PlayerID', 'Year'])
    for col in df.columns:
        # Cannot calculate rolling averages for these columns
        if col not in ['PlayerID', 'Year', 'Next_PPR', 'FantPos', 'Player']:
            # creating a new column for the rolling averages that we can add
            rolling_col = f'RollingAvg_{col}'
            # this creates the rolling average
            df[rolling_col] = df.groupby('PlayerID')[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
            # when there is no rolling average to compute, we fillna with the values of the column
            df[rolling_col] = df[rolling_col].fillna(df[col])
    return df

ffdata = add_rolling_averages(ffdata)


# Repeating the same process with these updated columns
train_data = ffdata.loc[ffdata['Year'] != 2021]
test_data = ffdata.loc[ffdata['Year'] == 2021]
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
X_train = train_data.drop(columns=['Next_PPR', 'PlayerID', 'Year', 'FantPos', 'Player'])
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

removed_columns = ['Next_PPR', 'PlayerID', 'Year', 'FantPos', 'Player']
selected_columns = train_data.columns[~train_data.columns.isin(removed_columns)]

# This is the target, what we want to predict
y_train = train_data['Next_PPR']

n_features_to_select = range(1, X_train.shape[1])

# List to hold cross-validation scores
cv_scores = []

rr1 = Ridge(alpha=1)

# Iterate over the number of features to select
for n in n_features_to_select:
    # This runs through and selects the optimal amount of features
    sfs = SequentialFeatureSelector(estimator=rr1, 
                                    n_features_to_select=n, 
                                    direction='forward', 
                                    scoring='neg_mean_squared_error', 
                                    cv=split)
    # Fitting the data with the sequential feature selector
    sfs.fit(X_train, y_train)
    # Cross-validate the model with the selected features
    score = np.mean(cross_val_score(rr1, sfs.transform(X_train), y_train, cv=split, scoring='neg_mean_squared_error'))
    cv_scores.append(score)

# Finding the optimal number of features
optimal_update = n_features_to_select[np.argmax(cv_scores)]
print("Optimal number of features:", optimal_update)
print("Best cross-validation neg MSE:", max(cv_scores))

# This is plotting the cross validation and mean squared error so we can visualize the scores and 
# amount of features
plt.figure(figsize=(10, 6))
plt.plot(n_features_to_select, cv_scores, marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation Negative MSE')
plt.title('Cross-Validation Negative MSE vs. Number of Features')
plt.grid()
plt.xticks(n_features_to_select)
plt.show()


# This creates the feature selector that selects the top 10 features from the data
sfs1 = SequentialFeatureSelector(rr1, n_features_to_select=10, direction='forward', cv=split)

# Fitting the sequential feature selector
sfs1.fit(X_train, y_train)

X_train_selected = sfs1.transform(X_train)

rr1.fit(X_train_selected, y_train)

# Getting the top 10 predictors
rolling_avg_predictors = selected_columns[sfs1.get_support()] 

# We end up adding new rolling average predictors to our model like the rank, games played
# rushing TD's, and PPR

X_test = test_data.drop(columns=['PlayerID', 'Year', 'FantPos', 'Player', 'Next_PPR'])
X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna() 
X_test_scaled = scaler.transform(X_test)
X_test_scaled = sfs1.transform(X_test_scaled)

y_pred = rr1.predict(X_test_scaled)

test_data['RollingAvg_Predicted_PPR'] = y_pred

# Evaluating how accurate our predictions were
mse_roll_avg = mean_squared_error(test_data['PPR'], test_data['RollingAvg_Predicted_PPR'])

# How do we know if this is any good?
# This gives us the standard deviation of 84.58
test_data['PPR'].describe()

# taking the square root
# this is lower than the standard deviation and lower than the original mse_sr, so this model is an improvement
# on our original model
mse_sr_rollavg = mse_roll_avg ** 0.5

# The MSE stands for mean squared error and it represents the difference between observed and predicted values
# Essentially, our predicted values are closer to the actual values for the second, rolling avg model

# Getting the difference between actual values and their predictions
test_data['roll_avg_diff'] = abs(test_data['PPR'] - test_data['RollingAvg_Predicted_PPR'])


# Now we can see how the rolling average model individually compares each RB to the original model
final_df = pd.merge(test_data_limit[['Player', 'diff']], test_data[['Player', 'roll_avg_diff']], on='Player')


# Future things to consider could be a players injury history, a team's draft or free agency acquisitions,
# or advanced metrics for running backs like EPA, yards after contact (YAC), or Elusive Rating (ELU)