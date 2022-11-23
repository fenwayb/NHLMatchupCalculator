import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pickle
import sqlalchemy
import sqlite3
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
# from splinter import Browser
# from bs4 import BeautifulSoup as soup
# from webdriver_manager.chrome import ChromeDriverManager

# home = "Boston Bruins"
# away = "New Jersey Devils"

def pull_all(home, away):
    # home = home
    # away = away
    # con = sqlite3.connect(':memory:')
    # cur = con.cursor()

    # cur.execute("CREATE TABLE home(pppctg, pkpctg, shots, shotsallowed, faceofpctg, shootingpctg, savepctg)")
    # cur.execute("CREATE TABLE away(pppctg, pkpctg, shots, shotsallowed, faceofpctg, shootingpctg, savepctg)")
    # cur.execute("CREATE TABLE results(results)")
    X = pd.read_csv('Resources/X.csv', index_col=1)
    team_info_df = pd.read_csv('resources/team_info.csv', index_col=1)
    team_info_df['teamName'] = team_info_df['shortName'] + " " + team_info_df['teamName']
    url = "https://statsapi.web.nhl.com/api/v1/teams"
    #scraping our own website for ids - is this the right way to do it?
    # idurl = "http://127.0.0.1:5000/"
    # executable_path = {'executable_path': ChromeDriverManager().install()}
    # browser = Browser('chrome', **executable_path, headless=True)
    def pull_home():
        # home = home
        home_id = team_info_df[team_info_df['teamName'] == f'{home}']
        home_name = home_id['teamName'].values[0]
        home_id = home_id['team_id'].values[0]
        home_url = f"{url}/{home_id}/?expand=team.stats"
        home_json = requests.get(home_url).json()
        homedata = []
        pppctg = home_json['teams'][0]['teamStats'][0]['splits'][0]['stat']['powerPlayPercentage']
        pkpctg = home_json['teams'][0]['teamStats'][0]['splits'][0]['stat']['penaltyKillPercentage']
        shots = home_json['teams'][0]['teamStats'][0]['splits'][0]['stat']['shotsPerGame']
        shotsallowed = home_json['teams'][0]['teamStats'][0]['splits'][0]['stat']['shotsAllowed']
        faceoffpctg = home_json['teams'][0]['teamStats'][0]['splits'][0]['stat']['faceOffWinPercentage']
        shootingpctg = home_json['teams'][0]['teamStats'][0]['splits'][0]['stat']['shootingPctg']
        savepctg = home_json['teams'][0]['teamStats'][0]['splits'][0]['stat']['savePctg']
        homedata.append({"pppctg_home" : float(pppctg)/100,
                "pkpctg_home" : float(pkpctg)/100,
                "shots_home" : shots,
                "shotsallowed_home" : shotsallowed,
                "faceoffpctg_home" : faceoffpctg,
                "shootingpctg_home" : float(shootingpctg)/100,
                "savepctg_home" : savepctg})
        home_df = pd.DataFrame(homedata)
        # cur.execute(f"INSERT INTO home VALUES ({list(homedata[0].values())})")
        # con.commit()
        return home_df, home_name
    home_df, home_name = pull_home(home)

    def pull_away():
        # away = away
        away_id = team_info_df[team_info_df['teamName'] == f"{away}"]
        away_name = away_id['teamName'].values[0]
        away_id = away_id['team_id'].values[0]
        away_url = f"{url}/{away_id}/?expand=team.stats"
        away_json = requests.get(away_url).json()
        awaydata = []
        pppctg = away_json['teams'][0]['teamStats'][0]['splits'][0]['stat']['powerPlayPercentage']
        pkpctg = away_json['teams'][0]['teamStats'][0]['splits'][0]['stat']['penaltyKillPercentage']
        shots = away_json['teams'][0]['teamStats'][0]['splits'][0]['stat']['shotsPerGame']
        shotsallowed = away_json['teams'][0]['teamStats'][0]['splits'][0]['stat']['shotsAllowed']
        faceoffpctg = away_json['teams'][0]['teamStats'][0]['splits'][0]['stat']['faceOffWinPercentage']
        shootingpctg = away_json['teams'][0]['teamStats'][0]['splits'][0]['stat']['shootingPctg']
        savepctg = away_json['teams'][0]['teamStats'][0]['splits'][0]['stat']['savePctg']
        awaydata.append({"pppctg_away" : float(pppctg)/100,
                "pkpctg_away" : float(pkpctg)/100,
                "shots_away" : shots,
                "shotsallowed_away" : shotsallowed,
                "faceoffpctg_away" : faceoffpctg,
                "shootingpctg_away" : float(shootingpctg)/100,
                "savepctg_away" : savepctg})
        away_df = pd.DataFrame(awaydata)
        # cur.execute(f"INSERT INTO away VALUES ({list(away[0].values())})")
        # con.commit()
        return away_df, away_name
    away_df, away_name = pull_away(away)
    
    compiled_stats_df = pd.concat([home_df,away_df],axis=1)
    scaler = MinMaxScaler().fit(X)
    compiled_scaled_df = pd.DataFrame(scaler.transform(compiled_stats_df),columns = compiled_stats_df.columns)
    with open('Resources/HockeyMLmodel.pkl', 'rb') as f:
        clf2 = pickle.load(f)
        results = clf2.predict(compiled_scaled_df[0:1])
        if results[0] == "0":
            results = away_name
        else:
            results = home_name
    
    # cur.execute(f"INSERT INTO results VALUES {results}")
    # con.commit()
    return home_df, away_df, results

home_df, away_df, results = pull_all("Boston Bruins", "New Jersey Devils")