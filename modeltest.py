# %%
# Segment Two Machine Learning Model
# Initial version by Josh Stowe, Team Hansen Brothers, Final Capstone Project 202211/8

# %%
# mlenv (Python 3.7.13)
# Import dependencies
import numpy as np
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from collections import Counter
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
# from imblearn.metrics import classification_report_imbalanced
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder


# %%
# # Load the game stats database for split, train, test
# file_path = Path('Resources/game_teams_stats.csv')
# df = pd.read_csv(file_path)
# df.head()
file_path = Path('Resources/ml_table.csv')
adv_df = pd.read_csv(file_path)
adv_df.head()


# %% [markdown]
# # Placeholder for preprocessing

# %%
adv_df.isna().sum()

# %%
# preprocessing, TBC after db cleaning complete
# merge dfs
# adv_df = advanced_home_df.reset_index().join(advanced_away_df, on='game_id', lsuffix='home_', rsuffix='away_')

# encode home_win true to 1, false to 0
# Use LabelEncoder to convert 'M/F' into integer labels
adv_df = adv_df
adv_df['won_home'] = LabelEncoder().fit_transform(adv_df['won_home'])

# drop columns not relevant
adv_df = adv_df.drop(columns="won_away")
adv_df = adv_df.drop(columns="abbreviation_home")
adv_df = adv_df.drop(columns="abbreviation_away")
adv_df = adv_df.drop(columns="team_id_home")
adv_df = adv_df.drop(columns="team_id_away")
adv_df = adv_df.drop(columns="game_id")
adv_df = adv_df.drop(columns="goals_home")
adv_df = adv_df.drop(columns="goals_away")


# encode winoutshootopp_home, winoutshotopp_home, winoutshotbyopp_away, winoutshootbyopp_away true to 1, false to 0
adv_df['winoutshootopp_home'] = LabelEncoder().fit_transform(adv_df['winoutshootopp_home'])
adv_df['winoutshotbyopp_home'] = LabelEncoder().fit_transform(adv_df['winoutshotbyopp_home'])
adv_df['winoutshootopp_away'] = LabelEncoder().fit_transform(adv_df['winoutshootopp_away'])
adv_df['winoutshotbyopp_away'] = LabelEncoder().fit_transform(adv_df['winoutshotbyopp_away'])

adv_df = adv_df.drop(columns="winoutshootopp_home")
adv_df = adv_df.drop(columns="winoutshotbyopp_home")
adv_df = adv_df.drop(columns="winoutshootopp_away")
adv_df = adv_df.drop(columns="winoutshotbyopp_away")


# %% [markdown]
# # Split the Data into Training and Testing

# %%
# Create the features
# adjust X df with the target features
X = adv_df.drop(columns="won_home")
X.head()

# %%
# create the target
y = adv_df["won_home"]
y.head()

# %%
X.describe()

# %%
X.dtypes

# %%
# Placeholder for column data type conversion

# %%
# Create train, test datasets
# Intention is to test sensitivity of test accuracy with the train/test size and select the split ratio that gives the 
# best accuracy prior to another sampling adjustments.
X_train, X_test, y_train, y_test = train_test_split(X, y)
# X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=None, test_size=0.15)



# %%
X_train

# %%
X_test

# %%
y_train

# %%
y_test

# %%
# Placeholder for any scaler implementation
# Scale the dataset using MinMaxScaler()
# X_scaled = MinMaxScaler().fit_transform(X)

# Scale the dataset using StandardScaler()
# X_scaled = StandardScaler().fit_transform(X)

# X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y,stratify=None, test_size=0.15)
# X_scaled

# %% [markdown]
# # ML model

# %%
# LogisticRegression model, predicting a 1 or 0 outcome
classifier = LogisticRegression()
classifier

classifier.fit(X_train, y_train)
# classifier.fit(X_train_scaled, y_train)

print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")

# print(f"Training Data Score: {classifier.score(X_train_scaled, y_train)}")
# print(f"Testing Data Score: {classifier.score(X_test_scaled, y_test)}")

predictions = classifier.predict(X_test)
pd.DataFrame({"Prediction": predictions, "Actual": y_test})

# Tesing the non-negative least squares if db allows
# model = LinearRegression(positive=True)

# # Fit the model to the training data, and calculate the scores for the training and testing data.
# model.fit(X_train, y_train)
# y_pred = model.fit(X_train,y_train).predict(X_test)
# y_pred = model.predict(X_test)

# training_score = model.score(X_train, y_train)
# testing_score = model.score(X_test, y_test)
# r2_score_model = r2_score(y_test,y_pred)

# print(f"Training Score: {training_score}")
# print(f"Testing Score: {testing_score}")
# print(f"R2 score", r2_score_model)

# %%


y_true = y_test
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

cm_df

# %%
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
accuracy = (tp + tn) / (tp + fp + tn + fn)
print(f"Accuracy: {accuracy}")

# %%
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=1, n_estimators=500).fit(X_train, y_train)
print(f'Training Score: {clf.score(X_train, y_train)}')
print(f'Testing Score: {clf.score(X_test, y_test)}')

# %%
from matplotlib import pyplot as plt
features = clf.feature_importances_
print(features)
plt.bar(x = range(len(features)), height=features)
plt.show()

# %%
precision = tp / (tp + fp)
precision

# %%
# Calculate the sensitivity of the model based on the confusion matrix
sensitivity = tp / (tp + fn)
sensitivity

# %%
# f1 = 2*precision*sensitivity / (precision + sensitivity)
# f1

# %%
# print(classification_report(y_true, y_pred))

# %%
# # Plot the residuals for the training and testing data.

# ### BEGIN SOLUTION
# plt.scatter(model.predict(X_train), model.predict(X_train) - y_train, c="blue", label="Training Data")
# plt.scatter(model.predict(X_test), model.predict(X_test) - y_test, c="orange", label="Testing Data")
# plt.legend()
# plt.hlines(y=0, xmin=y.min(), xmax=y.max())
# plt.title("Residual Plot")
# plt.show()
# ### END SOLUTION

# %%
# sharks then blues
X_prediction_test = pd.DataFrame([{0.186,0.923,31.0714,31.4286,49.6,0.083,0.895,0.207,0.708,29.0909,32.0909,51.4,0.072,0.878}])
X_prediction_test

# %%
prediction_test = classifier.predict(X_prediction_test)
pd.DataFrame({"Prediction": prediction_test})

# %%
# islanders then coyotes
X_prediction_test = pd.DataFrame([{0.191,0.870,31.7857,33.0,49.4,0.110,0.922,0.293,0.804,23.4167,36.6667,45.5,0.125,0.895}])
X_prediction_test

# %%
prediction_test = classifier.predict(X_prediction_test)
pd.DataFrame({"Prediction": prediction_test})

# %%
# # Dependencies
# import requests
# from pprint import pprint
# # from config import api_key

# query_url = "https://statsapi.web.nhl.com/api/v1/teams/1/stats"


# %%
# # Request articles
# # home_team = requests.get(query_url).json()
# home_team


# %%


# %%
# home_stats = home_team['stats']
# home_stats

# %%
# home_splits = home_stats[0]['splits']
# home_splits

# %%
# export model to h5


# %%
import requests

# %%
# home team
url = "https://statsapi.web.nhl.com/api/v1/teams/6/?expand=team.stats"
home = requests.get(url).json()


url = "https://statsapi.web.nhl.com/api/v1/teams/23/?expand=team.stats"
away = requests.get(url).json()


# %%
gamedata = []
id = home['teams'][0]['teamStats'][0]['splits'][0]['team']['id']
name = home['teams'][0]['teamStats'][0]['splits'][0]['team']['name']
gpg = home['teams'][0]['teamStats'][0]['splits'][0]['stat']['goalsPerGame']
gapg = home['teams'][0]['teamStats'][0]['splits'][0]['stat']['goalsAgainstPerGame']
pppctg = home['teams'][0]['teamStats'][0]['splits'][0]['stat']['powerPlayPercentage']
pkpctg = home['teams'][0]['teamStats'][0]['splits'][0]['stat']['penaltyKillPercentage']
shots = home['teams'][0]['teamStats'][0]['splits'][0]['stat']['shotsPerGame']
shotsallowed = home['teams'][0]['teamStats'][0]['splits'][0]['stat']['shotsAllowed']
winoutshootopp = home['teams'][0]['teamStats'][0]['splits'][0]['stat']['winOutshootOpp']
winoutshotbyopp = home['teams'][0]['teamStats'][0]['splits'][0]['stat']['winOutshotByOpp']
faceoffpctg = home['teams'][0]['teamStats'][0]['splits'][0]['stat']['faceOffWinPercentage']
shootingpctg = home['teams'][0]['teamStats'][0]['splits'][0]['stat']['shootingPctg']
savepctg = home['teams'][0]['teamStats'][0]['splits'][0]['stat']['savePctg']
gamedata.append({
    # "id" : id,
                  "name" : name,
                #  "gpg" : gpg,
                #  "gapg" : gapg,
                 "pppctg" : float(pppctg)/100,
                 "pkpctg" : float(pkpctg)/100,
                 "shots" : shots,
                 "shotsallowed" : shotsallowed,
                #  "winoutshootopp" : winoutshootopp,
                #  "winoutshotbyopp" : winoutshotbyopp,
                 "faceoffpctg" : faceoffpctg,
                 "shootingpctg" : float(shootingpctg)/100,
                 "savepctg" : savepctg})
home_df = pd.DataFrame(gamedata)

# %%
gamedata = []
id = away['teams'][0]['teamStats'][0]['splits'][0]['team']['id']
name = away['teams'][0]['teamStats'][0]['splits'][0]['team']['name']
gpg = away['teams'][0]['teamStats'][0]['splits'][0]['stat']['goalsPerGame']
gapg = away['teams'][0]['teamStats'][0]['splits'][0]['stat']['goalsAgainstPerGame']
pppctg = away['teams'][0]['teamStats'][0]['splits'][0]['stat']['powerPlayPercentage']
pkpctg = away['teams'][0]['teamStats'][0]['splits'][0]['stat']['penaltyKillPercentage']
shots = away['teams'][0]['teamStats'][0]['splits'][0]['stat']['shotsPerGame']
shotsallowed = away['teams'][0]['teamStats'][0]['splits'][0]['stat']['shotsAllowed']
winoutshootopp = away['teams'][0]['teamStats'][0]['splits'][0]['stat']['winOutshootOpp']
winoutshotbyopp = away['teams'][0]['teamStats'][0]['splits'][0]['stat']['winOutshotByOpp']
faceoffpctg = away['teams'][0]['teamStats'][0]['splits'][0]['stat']['faceOffWinPercentage']
shootingpctg = away['teams'][0]['teamStats'][0]['splits'][0]['stat']['shootingPctg']
savepctg = away['teams'][0]['teamStats'][0]['splits'][0]['stat']['savePctg']
gamedata.append({
    # "awayid" : id,
                 "awayname" : name,
                #  "gpg" : gpg,
                #  "gapg" : gapg,
                 "awaypppctg" : float(pppctg)/100,
                 "awaypkpctg" : float(pkpctg)/100,
                 "awayshots" : shots,
                 "awayshotsallowed" : shotsallowed,
                #  "winoutshootopp" : winoutshootopp,
                #  "winoutshotbyopp" : winoutshotbyopp,
                 "awayfaceoffpctg" : faceoffpctg,
                 "awayshootingpctg" : float(shootingpctg)/100,
                 "awaysavepctg" : savepctg})
away_df = pd.DataFrame(gamedata)

# %%
test_df = pd.concat([home_df,away_df],axis=1)
test_df = test_df.drop(columns="name")
test_df = test_df.drop(columns="awayname")
prediction_test = classifier.predict(test_df)
pd.DataFrame({"Prediction": prediction_test})


# %%
home_df

# %%
away_df

# %%



