'''
Aggregate dataframes collected from separate sources
'''
import pandas as pd 
import numpy as np 
import json
import pickle
from utils import doubleheaders, get_momentum_model_inputs, calculate_distance_model_inputs, id_converter, assemble_metrics
from gauss_rank_scaler import GaussRankScaler

print("Reading in statistics...")
try:

    today_games = pd.read_csv("./all_data/today_games.csv")
    
    season_totals = pd.read_csv("./all_data/season_totals.csv")

    starter_career = pd.read_csv("./all_data/starter_career.csv")

    starter_season = pd.read_csv("./all_data/starter_season.csv")

    all_starters = pd.read_csv("./all_data/past_raw.csv.gz", compression = "gzip")

except Exception as e:

    print("Could not read in statistics: {}".format(e))

    raise

s_map = id_converter(today_games, all_starters)

del all_starters

today_games["road_starter_"] = today_games.road_starter.map(s_map)

today_games["home_starter_"] = today_games.home_starter.map(s_map)

today_games = today_games.drop(columns = ["home_starter", "road_starter"]).rename(columns = {
    "road_starter_" : "road_starter",
    "home_starter_" : "home_starter"
})

working = pd.DataFrame(assemble_metrics(today_games, season_totals, starter_season, starter_career))

working = doubleheaders(working)

try: 

    working_elo = pd.read_csv("./all_data/daily_elo.csv.gz", compression = "gzip")

except Exception as e:

    print("Could not read in elo: {}".format(e))

    raise

working_elo["date"] = pd.to_datetime(working_elo.date, format = "%Y-%m-%d")

working_elo = doubleheaders(working_elo, home = "team1")

working_elo = working_elo.drop(columns = ["playoff", "neutral"]).rename(columns = {"team1" : "home_team",
                                                                     "team2" : "road_team"})

working_elo = working_elo.sort_index(ascending = False).reset_index(drop = True)

working_elo = working_elo[working_elo.date <= working.date.max()]

merge_cols1 = ["date", "home_team", "road_team", "is_doubleheader", "is_tripleheader"]

working = working.merge(working_elo, how = "left", left_on = merge_cols1,
                       right_on = merge_cols1)

drop_cols = ["elo1_pre", "elo2_pre", "elo_prob1", "elo_prob2", "elo1_post", "elo2_post",
            "pitcher1", "pitcher2", "rating_prob2", "rating1_post", "rating2_post", "year_x",
            "year_y"]

working = working.drop(drop_cols, axis = 1)

for_momentum = working_elo[working_elo.date < working.date.max()].reset_index(drop = True)

all_momentum = pd.DataFrame(get_momentum_model_inputs(working, for_momentum))

del for_momentum

merge_cols2 = ["date", "is_doubleheader", "is_tripleheader", "home_team", "road_team"]

working = working.merge(all_momentum, how = "left", left_on = merge_cols2, right_on = merge_cols2)

try:
    
    stadiums = pd.read_csv("./adv_metric_constants/all_stadiums_w_park_ids.csv", index_col = [0])

except Exception as e:

    print("Could not read in stadium data: {}".format(e))

    raise

stadiums = stadiums[["team_code", "year", "primary_latitude", "primary_longitude", "batting_park_factor"]]

stadiums = stadiums.rename(columns = {"year" : "season"})

working = working.merge(stadiums, how = "left", left_on = ["home_team", "season"],
                       right_on = ["team_code", "season"]).drop(columns = ["team_code", "primary_latitude",
                                                                            "primary_longitude"])

working_elo = working_elo.merge(stadiums, how = "left", left_on = ["season", "home_team"],
                       right_on = ["season" , "team_code"])

working_elo = working_elo.rename(columns = {"primary_latitude" : "home_latitude",
                                   "primary_longitude" : "home_longitude"}).drop("team_code", axis = 1)

working_elo = working_elo.merge(stadiums.drop("batting_park_factor", axis = 1), 
how = "left", left_on = ["season", "road_team"],
                       right_on = ["season", "team_code"])

working_elo = working_elo.rename(columns = {"primary_latitude" : "road_latitude",
                                   "primary_longitude" : "road_longitude"}).drop("team_code", axis = 1)

distance = pd.Series(calculate_distance_model_inputs(working, working_elo)).rename("distance_traveled")

working = pd.concat([working, distance], axis = 1)

try:

    odds = pd.read_csv("./all_data/current_odds.csv")

except Exception as e:

    print("Could not read in odds data: {}".format(e))

    raise

odds["date"] = pd.to_datetime(odds.date, format = "%Y-%m-%d")

odds = odds[["date", "is_doubleheader", "team1", "team2", "home_opening", 
"road_opening", "home_closing", "road_closing"]]

odds = odds.rename(columns = {
    "team1" : "home_team",
    "team2" : "road_team"
})

merge_cols3 = ["date", "is_doubleheader", "home_team", "road_team"]

working = working.merge(odds, how = "left", left_on = merge_cols3, right_on = merge_cols3).drop(columns = [
    "home_closing", "road_closing"
])

with open("./all_data/model_columns.json", "r+") as f:

    model_columns = json.load(f)

try:

    past_updated = pd.read_csv("./all_data/past_UPDATED.csv.gz", compression = "gzip")[model_columns]

except Exception as e:

    print("Could not read in scaler train data, {}".format(e))

model_ready = working[model_columns]

past_updated = pd.concat([past_updated, model_ready], axis = 0).reset_index(drop = True)

GRS = GaussRankScaler()

GRS.fit(past_updated)

model_ready = pd.DataFrame(GRS.transform(model_ready))

model_ready.columns = model_columns

today_games = doubleheaders(today_games, home = "home_team")

today_odds = doubleheaders(odds[odds.date == today_games.date.max()].reset_index(drop = True), home = "home_team")

merge_colsOdds = ["date", "year", "home_team", "road_team", "is_doubleheader", "is_tripleheader"]

today_games = today_games.merge(today_odds, how = "left",
left_on = merge_colsOdds,
right_on = merge_colsOdds).drop(columns = ["year", "is_doubleheader", "is_tripleheader", "home_opening", "road_opening"])

today_games.to_csv("./all_data/today_games.csv", index = False)

model_ready.to_csv("./all_data/model_prepared.csv", index = False)

print("Model ready data written to file")