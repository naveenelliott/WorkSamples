import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# accuracy means if you predict a win, what percentage of the time did the team actually win?
# if you predicted a loss, what percentage of the team did the team actually lose?
from sklearn.metrics import accuracy_score
# when we predicted a win, what percentage of the team did the team actually win?
from sklearn.metrics import precision_score

matches = pd.read_csv('RandomForestProject/matches.csv', index_col=0)

matches['date'] = pd.to_datetime(matches['date'])
matches['venue_code'] = matches['venue'].astype('category').cat.codes
matches['opp_code'] = matches['opponent'].astype('category').cat.codes
matches['hour'] = matches['time'].str.replace(":.+", "", regex=True).astype(int)
matches['day_code'] = matches['date'].dt.dayofweek

matches['target'] = (matches['result'] == 'W').astype(int)

# a random forest is a series of decision trees but each decision tree has slightly different parameters
# n_estimators is the number of individual decision trees we want to train - higher means longer run time but potentially more accuracy
# min_samples_split is the number of samples we want to have in a leaf of a decision tree before we split the node
# the higher this is, the more we are likely to overfit but there will be more accuracy, need to experiment to find optimal
# random_state - if we run the random_forest multiple times we will get the same result
rf = RandomForestClassifier(n_estimators=50, min_samples_split = 10, random_state=1)

# test set needs to come after training set
train = matches[matches['date'] < '2022-01-01']
test = matches[matches['date'] > '2022-01-01']

predictors = ['venue_code', 'opp_code', 'hour', 'day_code']

# trains random forest model with these predictors to predict the target
rf.fit(train[predictors], train['target'])

preds = rf.predict(test[predictors])

acc = accuracy_score(test['target'], preds)
#print(acc)
# when we predicted something would happen, 61.23% of the time, that thing actually happened

combined = pd.DataFrame(dict(actual=test['target'], prediction=preds))

#print(pd.crosstab(index=combined['actual'], columns=combined['prediction']))
# when we predicted a loss or a draw, most of the time we were correct
# when we predicted a win, we were wrong more often than we were right
# we need to revise accuracy metric, since we care about wins

#print(precision_score(test['target'], preds))
# when we predicted a win, the team won 47% of the time

grouped_matches = matches.groupby('team')
group = grouped_matches.get_group('Manchester City')

# Improving precision with rolling averages
def rolling_averages(group, cols, new_cols):
    group = group.sort_values('date')
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
new_cols = [f'{c}_rolling' for c in cols]
matches_rolling = matches.groupby('team').apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')

matches_rolling.index = range(matches_rolling.shape[0])

def make_predictions(data, predictors):
    train = data[data['date'] < '2022-01-01']
    test = data[data['date'] > '2022-01-01']
    rf.fit(train[predictors], train['target'])

    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test['target'], prediction=preds), index=test.index)
    precision = precision_score(test['target'], preds)
    return combined, precision

combined, precision = make_predictions(matches_rolling, predictors + new_cols)
#print(precision)
# Precision is improved from 47% to 62.5%

combined = pd.merge(combined, matches_rolling[['date', 'team', 'opponent', 'result']], left_index=True, right_index=True)

class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    'Brighton and Hove Albion': 'Brighton',
    'Manchester United': 'Manchester Utd',
    'Newcastle United': 'Newcastle Utd',
    'Tottenham Hotspur': 'Tottenham',
    'West Ham United': 'West Ham',
    'Wolverhampton Wanderers': 'Wolves'
    }

mapping = MissingDict(**map_values)
combined['new_team'] = combined['team'].map(mapping)

merged = combined.merge(combined, left_on=['date', 'new_team'], right_on=['date', 'opponent'])

temp = merged[(merged['prediction_x'] == 1) & (merged['prediction_y'] == 0)]['actual_x'].value_counts()
accuracy = 27 / 40
# accuracy is improved (67.5%), when we get both sides of the match and merge them together
# this is getting instances where one team was predicted to win, and another team was predicted to lose
# there are some instances where both teams are predicted to win, which is interesting

# next steps
# only two seasons worth of data, get 10 or 20 seasons worth of data to improve accuracy
# use more of the columns to generate predictions (venue, referee, captain, team record, and opponent record)
