import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#from pybaseball import batting_stats

# This is predicting wins above replacement for baseball players
batting = pd.read_csv('RandomForestProject/batting.csv')

batting = batting.groupby('IDfg', group_keys=False).filter(lambda x: x.shape[0] > 1)

def next_season(player):
    player = player.sort_values('Season')
    player['Next_WAR'] = player['WAR'].shift(-1)
    return player


batting = batting.groupby('IDfg', group_keys=False).apply(next_season)

null_count = batting.isnull().sum()

complete_cols = list(batting.columns[null_count == 0])

batting = batting[complete_cols + ['Next_WAR']].copy()

del batting['Dol']
del batting['Age Rng']
batting['team_code'] = batting['Team'].astype('category').cat.codes
batting_full = batting.copy()

batting = batting.dropna().copy()

rr = Ridge(alpha=1)
split = TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(rr, n_features_to_select=20, direction='forward', cv=split, n_jobs=4)
# want to select 20 features, will use time series split to do cross validation
# n_jobs makes things run faster by using multiple process cores
removed_columns = ['Next_WAR', 'Name', 'Team', 'IDfg', 'Season']
selected_columns = batting.columns[~batting.columns.isin(removed_columns)]

# When working with ridge regression, we need to scale the data
# MinMaxScaler scales the values to be between 0 and 1
scaler = MinMaxScaler()
batting.loc[:, selected_columns] = scaler.fit_transform(batting[selected_columns])


sfs.fit(batting[selected_columns], batting['Next_WAR'])
predictors = list(selected_columns[sfs.get_support()])

def backtest(data, model, predictors, start=5, step=1):
    all_predictions=[]
    years = sorted(data['Season'].unique())
    
    for i in range(start, len(years), step):
        current_year = years[i]
        
        train = data[data['Season'] < current_year]
        test = data[data['Season'] == current_year]
        
        model.fit(train[predictors], train['Next_WAR'])
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test['Next_WAR'], preds], axis=1)
        combined.columns = ['actual', 'prediction']
        
        all_predictions.append(combined)
    return pd.concat(all_predictions, axis=0)

predictions = backtest(batting, rr, predictors)    

# Evaluating how accurate our predictions were
mse = mean_squared_error(predictions['actual'], predictions['prediction'])

# How do we know if this is any good?
batting['Next_WAR'].describe()

# taking the square root
mse_sr = mse ** 0.5

# this is lower than the standard deviation but not significantly, so this model is ok, but can be made better


# we want to give the algorithm more information about how a player did during a previous season
def player_history(df):
    df = df.sort_values('Season')
    
    df['player_season'] = range(0, df.shape[0])
    # wins above replacement correlation from previous seasons combined to the next season
    # could look at this nonlinearly
    df['war_corr'] = list(df[['player_season', 'WAR']].expanding().corr().loc[(slice(None), 'player_season'), 'WAR'])
    df['war_corr'].fillna(1, inplace=True)
    
    df['war_diff'] = df['WAR']/df['WAR'].shift(1)
    df['war_diff'].fillna(1, inplace=True)
    
    df['war_diff'][df['war_diff'] == np.inf] = 1
    
    return df

batting = batting.groupby('IDfg', group_keys=False).apply(player_history)


def group_averages(df):
    return df['WAR'] / df['WAR'].mean()

batting['war_season'] = batting.groupby('Season', group_keys=False).apply(group_averages)

new_predictors = predictors + ['player_season', 'war_corr', 'war_season', 'war_diff']
predictions = backtest(batting, rr, new_predictors)

mse_update = mean_squared_error(predictions['actual'], predictions['prediction'])
# Better than before, we have improved with our new predictors

significant_predictors = pd.Series(rr.coef_, index=new_predictors).sort_values()

diff = predictions['actual'] - predictions['prediction']
merged = predictions.merge(batting, left_index=True, right_index=True)
merged['diff'] = abs(predictions['actual'] - predictions['prediction'])

merged = merged[['IDfg', 'Season', 'Name', 'WAR', 'Next_WAR', 'diff']].sort_values(['diff'], ascending=False)

# Figure out how to handle injuries as a next step
