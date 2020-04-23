'''
Prepare raw data scraped from Baseball-Reference.com for model
'''
import pandas as pd 
import numpy as np 
import json
import os 
import re
from bs4 import BeautifulSoup
import requests

def prepare_data():

    team_map = {
        
        "CincinnatiReds" : "CIN",
        "KansasCityRoyals" : "KCR",
        "LosAngelesDodgers" : "LAD",
        "MiamiMarlins" : "FLA",
        "MilwaukeeBrewers" : "MIL",
        "MinnesotaTwins" : "MIN",
        "NewYorkYankees" : "NYY",
        "OaklandAthletics" : "OAK",
        "PhiladelphiaPhillies" : "PHI",
        "SanDiegoPadres" : "SDP",
        "SeattleMariners" : "SEA",
        "TampaBayRays" : "TBD",
        "TexasRangers" : "TEX",
        "TorontoBlueJays" : "TOR",
        "WashingtonNationals" : "WSN",
        "AtlantaBraves" : "ATL",
        "ClevelandIndians" : "CLE",
        "PittsburghPirates" : "PIT",
        "LosAngelesAngels" : "ANA",
        "BaltimoreOrioles" : "BAL",
        "DetroitTigers" : "DET",
        "NewYorkMets" : "NYM",
        "ArizonaDiamondbacks" : "ARI",
        "ChicagoWhiteSox" : "CHW",
        "ColoradoRockies" : "COL",
        "HoustonAstros" : "HOU",
        "SanFranciscoGiants" : "SFG",
        "StLouisCardinals" : "STL",
        "BostonRedSox" : "BOS",
        "ChicagoCubs" : "CHC"
    }

    current_batting = pd.read_csv("./all_data/current_batting.csv")

    current_pitching = pd.read_csv("./all_data/current_pitching.csv")

    dfs = [current_batting, current_pitching]

    updated_frames = []

    for df in dfs:
        
        df["date"] = pd.to_datetime(df.date, format = "%Y-%m-%d")
        
        df['year'] = pd.DatetimeIndex(df.date).year
        
        df = df.assign(is_doubleheader = 0, is_tripleheader = 0)
        
        game_counts = df.groupby('home').date.value_counts()

        double_headers = game_counts[game_counts == 2]
        triple_headers = game_counts[game_counts > 2]

        all_double_headers = []
        for j in double_headers.index:
            all_double_headers.append(j)

        all_triple_headers = []
        for k in triple_headers.index:
            all_triple_headers.append(k)

        for index in all_double_headers:
            game_indices = df[(df.team_code == index[0]) & (df.date == index[1])].index
            if len(game_indices) > 1:
                df.at[game_indices[1], 'is_doubleheader'] = 1
            else:
                print(index)

        for index_ in all_triple_headers:
            game_indices_ = df[(df.team_code == index_[0]) & (df.date == index_[1])].index
            if len(game_indices_) == 3:
                df.at[game_indices_[1], 'is_doubleheader'] = 1
                df.at[game_indices_[2], 'is_tripleheader'] = 1
            else:
                print(index_)
                
        updated_frames.append(df)

    merge_cols = ["date", "visitor", "home", "is_doubleheader", "is_tripleheader", "year"]

    full_frame = updated_frames[0].merge(updated_frames[1], how = "left", left_on = merge_cols,
                                        right_on = merge_cols)

    full_frame["home_team"] = full_frame.home.map(team_map)

    full_frame["road_team"] = full_frame.visitor.map(team_map)

    full_frame = full_frame.drop(columns = ["home", "visitor"])

    full_frame = full_frame.drop(columns = ["visitorstarter", "homestarter"])

    full_frame = full_frame.rename({"homeretrosheet_id" : "home_starter",
                                "visitorretrosheet_id" : "road_starter"}, axis = 1)

    all_cols = list(full_frame.columns)

    updated_cols = []

    for col in all_cols:
        
        col = col.replace("visitorstarter", "road_starter_")
        
        col = col.replace("visitorbullpen", "road_relief_")
        
        col = col.replace("homestarter", "home_starter_")
        
        col = col.replace("homebullpen", "home_relief_")
        
        if "starter" not in col or "relief" not in col:
            
            col = col.replace("home", "home_")
            
            col = col.replace("visitor", "road_")
            
        col = col.replace("__", "_")
        
        col = col.replace("SO", "K")
        
        updated_cols.append(col) 

    full_frame.columns = updated_cols

    prefix = ["home_", "road_"]

    for pre in prefix:
        
        full_frame[pre + "TB"] = (full_frame[pre + "HR"] * 4) + (full_frame[pre + "3B"] * 3) +\
        (full_frame[pre + "2B"] * 2) + full_frame[pre + "1B"]

    full_frame = full_frame.rename(columns = {"year" : "season"})

    prefix = ["home_", "road_", "home_starter_", "home_relief_", "road_starter_", "road_relief_"]

    for pre in prefix:
        
        full_frame[pre + "SAC"] = full_frame[pre + "SF"] + full_frame[pre + "SH"]
        

    full_frame = full_frame.drop(columns = ["home_SF", "home_SH", "road_SF", "road_SH",
                                        "home_starter_SF", "home_starter_SH", "home_relief_SF",
                                        "home_relief_SH", "road_starter_SF", "road_starter_SH", "road_relief_SH",
                                        "road_relief_SF"])

    cols = ["home_starter_IP", "road_starter_IP", "home_relief_IP", "road_relief_IP"]

    for col in cols:
        
        full_frame[col] = full_frame[col].astype("str")
        
        for k in range(len(full_frame)):
            
            full_frame.at[k, col] = full_frame.iloc[k][col].replace(".1", ".33")
            
            full_frame.at[k, col] = full_frame.iloc[k][col].replace(".2", ".67")
            
            full_frame.at[k, col] = full_frame.iloc[k][col].replace(".8", ".33")
            
            full_frame.at[k, col] = full_frame.iloc[k][col].replace(".9", ".67")
            
        full_frame[col] = full_frame[col].astype("float32")

    prefix = ["home_starter_", "road_starter_", "home_relief_", "road_relief_"]

    for pre in prefix:
        
        full_frame[pre + "AB"] = full_frame[pre + "H"] + (full_frame[pre + "IP"] * 3).astype("int32")
        
        full_frame[pre + "PA"] = (full_frame[pre + "IP"] * 3).astype("int32") + full_frame[pre + "H"] +\
        full_frame[pre + "BB"] + full_frame[pre + "IBB"] + full_frame[pre + "HBP"] +\
        full_frame[pre + "SAC"]

    prefix = ["home_starter_", "road_starter_", "home_relief_", "road_relief_"]

    for pre in prefix:
        
        full_frame[pre + "1B"] = full_frame[pre + "H"] - full_frame[pre + "2B"] + full_frame[pre + "3B"] +\
        full_frame[pre + "HR"]

    with open("./adv_metric_constants/wOBA_weights.json", "r+") as f:
        
        wOBA_weights = json.load(f)

    wOBA = pd.DataFrame(wOBA_weights)

    change_cols = ["wOBA", "wOBAScale", "wBB", "wHBP", "w1B", "w2B", "w3B", "wHR",
                "runSB", "runCS", "R/PA", "R/W", "cFIP"]

    wOBA["Season"] = wOBA.Season.astype("int64")

    for col in change_cols:
        
        wOBA[col] = wOBA[col].astype("float32")
        
    wOBA = wOBA.rename(columns = {"Season" : "season"})

    full_frame = full_frame.merge(wOBA, how = "left", left_on = ["season"],
                            right_on = ["season"])

    teams = requests.get("https://www.retrosheet.org/TEAMABR.TXT").content

    team_soup = BeautifulSoup(teams, "html.parser").get_text().split("\n")

    leagues = {}

    pattern = r'([\w]+)'

    for row in team_soup:
            
        vals = re.findall(pattern, row)
        
        try:
        
            leagues[vals[0]] = vals[1]
            
        except:
            
            continue

    with open("./adv_metric_constants/modern_rc.json", "r+") as f:
        retrosheet_codes = json.load(f)
        
    retrosheet_codes.update({"MIA" : "FLA"})

    pop_list = []

    for key in leagues:
        
        if key not in retrosheet_codes:
            
            pop_list.append(key)

    for key in pop_list:
        
        leagues.pop(key)

    leagues["MIA"] = "NL"

    df = pd.DataFrame({"team_code" : list(leagues.keys()), "league" : list(leagues.values())})

    df["elo_code"] = df.team_code.map(retrosheet_codes)

    elo_leagues = {}

    for key in retrosheet_codes:
        
        elo_leagues[retrosheet_codes[key]] = leagues[key]

    elo_leagues["HOU"] = "AL"

    full_frame["home_league"] = full_frame.home_team.map(elo_leagues)

    full_frame["road_league"] = full_frame.road_team.map(elo_leagues)

    al_league_wRC = pd.read_csv("./adv_metric_constants/league_wRC_AL.csv")

    nl_league_wRC = pd.read_csv("./adv_metric_constants/league_wRC_NL.csv")

    al_league_wRC = al_league_wRC[["Season", "PA", "wRC"]].rename(columns = {
        "Season" : "season",
        "PA" : "league_PA",
        "wRC" : "league_wRC"
    })

    nl_league_wRC = nl_league_wRC[["Season", "PA", "wRC"]].rename(columns = {
        "Season" : "season",
        "PA" : "league_PA",
        "wRC" : "league_wRC"
    })

    al_league_wRC = al_league_wRC.assign(league = "AL")

    nl_league_wRC = nl_league_wRC.assign(league = "NL")

    league_wRC = pd.concat([al_league_wRC, nl_league_wRC], axis = 0).sort_values(by = ["season"]).reset_index(drop = True)

    full_frame = full_frame.merge(league_wRC, how = "left", left_on = ["season", "home_league"],
                        right_on = ["season", "league"])

    full_frame = full_frame.drop(columns = ["league"]).rename(columns = {
        "league_PA" : "home_league_PA",
        "league_wRC" : "home_league_wRC"
    }
    )

    full_frame = full_frame.merge(league_wRC, how = "left", left_on = ["season", "road_league"],
                        right_on = ["season", "league"])

    full_frame = full_frame.drop(columns = ["league"]).rename(columns = {
        "league_PA" : "road_league_PA",
        "league_wRC" : "road_league_wRC"
    })

    all_stadiums = pd.read_csv("./adv_metric_constants/all_stadiums_w_park_ids.csv", index_col = 0)

    park_factors = all_stadiums[["team_code", "year", "batting_park_factor"]]

    full_frame = full_frame.merge(park_factors, how = "left", 
                            left_on = ["home_team", "season"],
                            right_on = ["team_code", "year"])

    full_frame = full_frame.drop(columns = ["team_code", "year"])

    full_frame = full_frame.rename(columns = {"batting_park_factor" : "home_batting_park_factor"})

    full_frame.home_batting_park_factor = full_frame.home_batting_park_factor / 100.

    full_frame.to_csv("./all_data/current_season.csv", index = False)

if __name__ == "__main__":

    prepare_data()



