'''
Collection of utility functions used in data collection and data manipulation
'''

import pandas as pd 
import numpy as np 

def doubleheaders(df, home = "home_team"):
    df["date"] = pd.to_datetime(df.date, format = "%Y-%m-%d")
        
    df['year'] = pd.DatetimeIndex(df.date).year

    df = df.assign(is_doubleheader = 0, is_tripleheader = 0)

    game_counts = df.groupby(home).date.value_counts()

    double_headers = game_counts[game_counts == 2]
    triple_headers = game_counts[game_counts > 2]

    all_double_headers = []
    for j in double_headers.index:
        all_double_headers.append(j)

    all_triple_headers = []
    for k in triple_headers.index:
        all_triple_headers.append(k)

    for index in all_double_headers:
        game_indices = df[(df[home] == index[0]) & (df.date == index[1])].index
        if len(game_indices) > 1:
            df.at[game_indices[1], 'is_doubleheader'] = 1
        else:
            print(index)

    for index_ in all_triple_headers:
        game_indices_ = df[(df[home] == index_[0]) & (df.date == index_[1])].index
        if len(game_indices_) == 3:
            df.at[game_indices_[1], 'is_doubleheader'] = 1
            df.at[game_indices_[2], 'is_tripleheader'] = 1
        else:
            print(index_)

    return(df)

def get_momentum(df, k):

    home_team, road_team = df.iloc[k]["home_team"], df.iloc[k]["road_team"]

    d = {}

    keep_cols = ["date", "is_doubleheader", "is_tripleheader", "season", "home_team", "road_team"]

    for col in keep_cols:

        d[col] = df.iloc[k][col]

    home_full = df[(df.home_team == home_team) | (df.road_team == home_team)].reset_index(drop = True)

    road_full = df[(df.home_team == road_team) | (df.road_team == road_team)].reset_index(drop = True)

    home_df = df[df.home_team == home_team].reset_index(drop = True)

    road_df = df[df.road_team == road_team].reset_index(drop = True)

    if len(home_df) == 0:

        d["home_record"] = 0

        d["run_differential_hm"] = 0

        d["avg_margin_hm"] = 0

        d["home_H_streak"] = 0

    else:

        d["home_record"] = len(home_df[home_df.score1 > home_df.score2]) - len(home_df[home_df.score2 > home_df.score1])

        d["run_differential_hm"] = home_df.score1.sum() - home_df.score2.sum()

        d["avg_margin_hm"] = d["run_differential_hm"] / len(home_df)

        d["home_H_streak"] = monitor_streak(home_team, home_df, style = "home")

    if len(home_full) == 0:

        d["home_streak"] = 0

    else:

        d["home_streak"] = monitor_streak(home_team, home_full, style = "overall")

    if len(road_df) == 0:

        d["road_record"] = 0

        d["run_differential_rd"] = 0

        d["avg_margin_rd"] = 0

        d["road_R_streak"] = 0

    else:

        d["road_record"] = len(road_df[road_df.score2 > road_df.score1]) - len(road_df[road_df.score1 > road_df.score2])

        d["run_differential_rd"] = road_df.score2.sum() - road_df.score1.sum()

        d["avg_margin_rd"] = d["run_differential_rd"] / len(road_df)

        d["road_R_streak"] = monitor_streak(road_team, road_df, style = "road")

    if len(road_full) == 0:

        d["road_streak"] = 0

    else:

        d["road_streak"] = monitor_streak(road_team, road_full, style = "overall")

    return(d)

def monitor_streak(team, df, style:str = "overall"):

    if len(df) == 0:

        return(0)

    df["home_win"] = (df.score1 > df.score2).astype("uint8")

    k = -1

    streak = 0

    last_game = "N"

    if style == "overall":

        while k >= -len(df):
    
            if df.iloc[k].home_team == team and df.iloc[k].home_win == 1:
                
                if last_game == "W" or last_game == "N":
                    
                    streak += 1
                    
                    k -= 1
                    
                    last_game = "W"
                    
                else:
                    
                    break
                    
            elif df.iloc[k].home_team == team and df.iloc[k].home_win == 0:
                
                if last_game == "L" or last_game == "N":
                    
                    streak -= 1
                    
                    k -= 1
                    
                    last_game = "L"
                    
                else:
                    
                    break
                    
            elif df.iloc[k].road_team == team and df.iloc[k].home_win == 0:
                
                if last_game == "W" or last_game == "N":
                    
                    streak += 1
                    
                    k -= 1
                    
                    last_game = "W"
                    
                else:
                    
                    break
                    
            else:
                
                if last_game == "L" or last_game == "N":
                    
                    streak -= 1
                    
                    k -= 1
                    
                    last_game = "L"
                    
                else:
                    
                    break

    elif style == "home":

        while k >= -len(df):

            if df.iloc[k]["home_win"] == 1:

                if last_game == "N" or last_game == "W":

                    streak += 1

                    k -= 1

                    last_game == "W"

                else:

                    break

            else:

                if last_game == "N" or last_game == "L":

                    streak -= 1

                    k -= 1

                    last_game == "L"

                else:

                    break

    else:

        while k >= -len(df):
        
            if df.iloc[k]["home_win"] == 0:

                if last_game == "N" or last_game == "W":

                    streak += 1

                    k -= 1

                    last_game == "W"

                else:

                    break

            else:

                if last_game == "N" or last_game == "L":

                    streak -= 1

                    k -= 1

                    last_game == "L"

                else:

                    break

    return(streak)

def calculate_distance(working):

    memory = {}

    d_t = []

    for k in range(len(working)):
        
        road_team = working.iloc[k]["road_team"]
        
        home_team = working.iloc[k]["home_team"]
        
        if road_team not in memory:
            
            memory[road_team] = {}
            
            memory[road_team]["current_opponent"] = home_team
            
            memory[road_team]["on_roadtrip"] = True
            
            memory[road_team]["distance_traveled"] = haversine_distance(working.iloc[k]["home_latitude"],
                                                                        working.iloc[k]["home_longitude"],
                                                                        working.iloc[k]["road_latitude"],
                                                                        working.iloc[k]["road_longitude"])
            d_t.append(memory[road_team]["distance_traveled"])
            
        else:
            
            if memory[road_team]["current_opponent"] == home_team:
                
                d_t.append(memory[road_team]["distance_traveled"])
                
            elif memory[road_team]["on_roadtrip"]:
                
                memory[road_team]["current_opponent"] = home_team
                
                addtl_distance = haversine_distance(working.iloc[k]["home_latitude"],
                                                    working.iloc[k]["home_longitude"],
                                                    working.iloc[k]["road_latitude"],
                                                    working.iloc[k]["road_longitude"])
                
                memory[road_team]["distance_traveled"] += addtl_distance
                
                d_t.append(memory[road_team]["distance_traveled"])
                
            else:
            
                memory[road_team]["current_opponent"] = home_team
            
                memory[road_team]["on_roadtrip"] = True

                memory[road_team]["distance_traveled"] = haversine_distance(working.iloc[k]["home_latitude"],
                                                                            working.iloc[k]["home_longitude"],
                                                                            working.iloc[k]["road_latitude"],
                                                                            working.iloc[k]["road_longitude"])
                d_t.append(memory[road_team]["distance_traveled"])
                
        if home_team not in memory:
            
            memory[home_team] = {}
            
        memory[home_team]["current_opponent"] = road_team

        memory[home_team]["on_roadtrip"] = False

        memory[home_team]["distance_traveled"] = 0

    return(d_t)

def haversine_distance(latitude_1, longitude_1, latitude_2, longitude_2):
    R = 6378.137
    h = np.arcsin( np.sqrt(np.sin( (np.radians(latitude_2) - np.radians(latitude_1))/2)**2 \
                        + np.cos(np.radians(latitude_1))*np.cos(np.radians(latitude_2))*\
                        np.sin( (np.radians(longitude_2) - np.radians(longitude_1))/2)**2))
    return(2 * R * h)

def most_recent(df, teams, split_inv, home_only, road_only):

    all_teams = list(df.home_team.unique())
    
    vals = []
    
    for team in all_teams:
        
        d = {}
        
        full = df[(df.home_team == team) | (df.road_team == team)]
        
        road = df[df.road_team == team]
        
        home = df[df.home_team == team]
        
        d["team"] = team
        
        if full.iloc[-1]["home_team"] == team:
    
            splits = ["home" + i for i in split_inv]

            stats = full.iloc[-1][splits]
            
            stats.index = split_inv
            
            d.update(stats.to_dict())

            d.update(full.iloc[-1][home_only].to_dict())

            d.update(road.iloc[-1][road_only].to_dict())
            
        elif full.iloc[-1]["road_team"] == team:
            
            splits = ["road" + i for i in split_inv]
            
            stats = full.iloc[-1][splits]
            
            stats.index = split_inv
            
            d.update(stats.to_dict())
            
            d.update(full.iloc[-1][road_only].to_dict())
            
            d.update(home.iloc[-1][home_only].to_dict())
            
        vals.append(d)
        
    updated_frame = pd.DataFrame(vals)
    
    updated_frame.to_csv("./all_data/model_updated.csv", index = False)

team_cols = ['_AB',
 '_H',
 '_PA',
 '_BB',
 '_R',
 '_HR',
 '_2B',
 '_3B',
 '_IBB',
 '_HBP',
 '_1B',
 '_relief_IP',
 '_relief_H',
 '_relief_ER',
 '_relief_BB',
 '_relief_K',
 '_relief_HR',
 '_relief_2B',
 '_relief_3B',
 '_relief_IBB',
 '_relief_HBP',
 '_TB',
 '_SAC',
 '_relief_SAC',
 '_relief_AB',
 '_relief_PA',
 '_relief_1B',
]

constants = [
    "wOBA", "wOBAScale", "wBB", "wHBP", "w1B", "w2B", "w3B", "wHR", "R/PA", "cFIP"
]

def aggregate(current_season, team_cols = team_cols, constants = constants):

    all_vals = []

    for team in current_season.home_team.unique():
        
        d = {}
        
        d["team"] = team
        
        for col in team_cols:
            
            d[col] = current_season[current_season.home_team == team]["home" + col].sum() +\
            current_season[current_season.road_team == team]["road" + col].sum()
            
        d["batting_park_factor"] = current_season[(current_season.home_team == team) |
                                                (current_season.road_team == team)].home_batting_park_factor.mean()
        
        d["league_PA"] = current_season[current_season.home_team == team].home_league_PA.max()
        
        d["league_wRC"] = current_season[current_season.home_team == team].home_league_wRC.max()
        
        for col in constants:
            
            d[col] = current_season[(current_season.home_team == team)][col].mean()
            
        all_vals.append(d)

    df = pd.DataFrame(all_vals)

    return(df)
    
def id_converter(current_season, career_data):
    
    all_current = set(list(current_season.home_starter.unique()) + list(current_season.road_starter.unique()))
    
    all_time = set(list(career_data.home_starter.unique()) + list(career_data.road_starter.unique()))

    s_map = {}

    for starter in all_current:

        if len(starter) == 9:
            
            n = starter[-1]

            retro_id = starter[:4] + starter[5] + "00" + n
            
            if retro_id not in all_time:
                
                while int(n) < 10:
                    
                    n = str(int(n) + 1)
                    
                    retro_id = starter[:4] + starter[5] + "00" + n
                    
                    if retro_id in all_time:
                        
                        break

            s_map[starter] = retro_id

        elif len(starter) == 8:

            retro_id = starter[:4] + starter[4] + "00" + starter[-1]
            
            if retro_id not in all_time:
                
                while int(n) < 10:
                    
                    n = str(int(n) + 1)
                    
                    retro_id = starter[:4] + starter[4] + "00" + n
                    
                    if retro_id in all_time:
                        
                        break

            s_map[starter] = retro_id

        elif len(starter) == 7:

            retro_id == starter[:3] + "-" + starter[3] + "00" + starter[-1]
            
            if retro_id not in all_time:
                
                while int(n) < 10:
                    
                    n = str(int(n) + 1)
                    
                    retro_id = starter[:3] + "-"  + starter[3] + "00" + n
                    
                    if retro_id in all_time:
                        
                        break

            s_map[starter] = retro_id
            
    return(s_map)

starter_cols = {'_starter_1B',
 '_starter_2B',
 '_starter_3B',
 '_starter_AB',
 '_starter_BB',
 '_starter_ER',
 '_starter_H',
 '_starter_HBP',
 '_starter_HR',
 '_starter_IBB',
 '_starter_IP',
 '_starter_K',
 '_starter_PA',
 '_starter_SAC'}

wOBA_weight_cols = ["wBB", "wHBP", "w1B", "w2B", "w3B", "wHR"]

def aggregate_starter_career(current_season, career_data,
  starter_cols = starter_cols, wOBA_weight_cols = wOBA_weight_cols):
    
    def calc_weighted_mean(df, starter, stat):

        home_mean = df[df.home_starter == starter]["home_" + stat].mean()

        home_scalar = len(df[df.home_starter == starter])

        road_mean = df[df.road_starter == starter]["road_" + stat].mean()

        road_scalar = len(df[df.road_starter == starter])

        return(
        (home_mean * (home_scalar / (home_scalar + road_scalar))) +
            (road_mean * (road_scalar / (home_scalar + road_scalar)))
        )

    s_map = id_converter(current_season, career_data)
    
    current_season["home_starter_"] = current_season.home_starter.map(s_map)

    current_season["road_starter_"] = current_season.road_starter.map(s_map)

    current_season = current_season.drop(columns = ["home_starter", "road_starter"]).rename(
                                        columns = {"home_starter_" : "home_starter",
                                                    "road_starter_" : "road_starter"})

    all_career = pd.concat([career_data, current_season], axis = 0, sort = True).reset_index(drop = True)

    all_starters = set(list(all_career.road_starter.unique()) + list(all_career.home_starter.unique()))

    all_vals = []

    for starter in all_starters:

        d = {}

        d["starter"] = starter

        for col in starter_cols:

            d[col] = all_career[all_career.home_starter == starter]["home" + col].sum() +\
            all_career[all_career.road_starter == starter]["road" + col].sum()

        if len(all_career[all_career.home_starter == starter]) !=0 and \
        len(all_career[all_career.road_starter == starter]) != 0:

            d["league_PA"] = calc_weighted_mean(all_career, starter, "league_PA")

            d["league_wRC"] = calc_weighted_mean(all_career, starter, "league_wRC")

        elif len(all_career[all_career.home_starter == starter]) != 0:

            d["league_PA"] = all_career[all_career.home_starter == starter].home_league_PA.mean()

            d["league_wRC"] = all_career[all_career.home_starter == starter].home_league_wRC.mean()

        else:

            d["league_PA"] = all_career[all_career.road_starter == starter].road_league_PA.mean()

            d["league_wRC"] = all_career[(all_career.road_starter == starter)].road_league_wRC.mean()

        career_starts = all_career[(all_career.home_starter == starter) |
                                    (all_career.road_starter == starter)]

        d["n_starts"] = len(career_starts)

        d["batting_park_factor"] = career_starts.home_batting_park_factor.mean()

        for col in wOBA_weight_cols:

            d[col] = career_starts[col].mean()

        d["RPA"] = career_starts["R/PA"].mean()

        d["wOBA_constant"] = career_starts.wOBA.mean()

        d["wOBA_scale"] = career_starts.wOBAScale.mean()

        d["FIP_constant"] = career_starts.cFIP.mean()

        all_vals.append(d)

    df = pd.DataFrame(all_vals)  
    
    return(df)

def aggregate_starter_season(current_season, career_data, 
starter_cols = starter_cols, wOBA_weight_cols = wOBA_weight_cols):
    
    s_map = id_converter(current_season, career_data)
    
    current_season["home_starter_"] = current_season.home_starter.map(s_map)

    current_season["road_starter_"] = current_season.road_starter.map(s_map)

    current_season = current_season.drop(columns = ["home_starter", "road_starter"]).rename(
                                        columns = {"home_starter_" : "home_starter",
                                                    "road_starter_" : "road_starter"})
    
    all_starters = set(list(current_season.home_starter.unique()) + list(current_season.road_starter.unique()))
    
    all_vals = []

    for starter in all_starters:

        d = {}

        d["starter"] = starter

        for col in starter_cols:

            d[col] = current_season[current_season.home_starter == starter]["home" + col].sum() +\
            current_season[current_season.road_starter == starter]["road" + col].sum()

        if len(current_season[current_season.home_starter == starter]) !=0:

            d["league_PA"] = current_season[current_season.home_starter == starter].home_league_PA.mean()

            d["league_wRC"] = current_season[current_season.home_starter == starter].home_league_wRC.mean()

        else:

            d["league_PA"] = current_season[current_season.road_starter == starter].road_league_PA.mean()

            d["league_wRC"] = current_season[(current_season.road_starter == starter)].road_league_wRC.mean()

        season_starts = current_season[(current_season.home_starter == starter) |
                                    (current_season.road_starter == starter)]

        d["n_starts"] = len(season_starts)

        d["batting_park_factor"] = season_starts.home_batting_park_factor.mean()

        for col in wOBA_weight_cols:

            d[col] = season_starts[col].mean()

        d["RPA"] = season_starts["R/PA"].mean()

        d["wOBA_constant"] = season_starts.wOBA.mean()

        d["wOBA_scale"] = season_starts.wOBAScale.mean()

        d["FIP_constant"] = season_starts.cFIP.mean()

        all_vals.append(d)

    df = pd.DataFrame(all_vals)  
    
    return(df)


team_metrics = ["_wOBA","_relief_wOBA", "_wRAA", "_relief_wRAA", "_wRC",
        "_relief_wRC","_OPS", "_relief_FIP", "_relief_WHIP", "_relief_ERA", "_relief_K_9",
        "_relief_K_BB"]

starter_season_metrics = ["_starter_season_wOBA", "_starter_season_wRAA", "_starter_season_wRC",
                          "_starter_season_FIP", "_starter_season_WHIP", "_starter_season_ERA",
                          "_starter_seasonK/9", "_starter_seasonK/BB", "_starter_seasonAVG_IP"]

starter_career_metrics = ["_starter_career_wOBA", "_starter_career_wRAA", "_starter_career_wRC",
                          "_starter_career_FIP", "_starter_career_WHIP", "_starter_career_ERA",
                          "_starter_careerK/9", "_starter_careerK/BB", "_starter_career_AVGIP"]

def assemble_metrics(today_games, season_totals, starter_season, starter_career,
                    team_metrics = team_metrics,
                    starter_season_metrics = starter_season_metrics,
                    starter_career_metrics = starter_career_metrics):
    
    all_inputs = []
    
    for k in range(len(today_games)):
        
        di = {}
        
        di["date"] = today_games.iloc[k]["date"]
        
        di["home_team"] = today_games.iloc[k]["home_team"]
        
        di["road_team"] = today_games.iloc[k]["road_team"]
        
        road_stats = season_totals[season_totals.team == today_games.iloc[k]["road_team"]][team_metrics].values[0]
        
        home_stats = season_totals[season_totals.team == today_games.iloc[k]["home_team"]][team_metrics].values[0]
        
        if len(starter_season[starter_season.starter == today_games.iloc[k]["road_starter"]]) == 0:
            
            road_starter_season = np.zeros(len(starter_season_metrics))
            
        else:
        
            road_starter_season = starter_season[starter_season.starter == \
                                                 today_games.iloc[k]["road_starter"]][starter_season_metrics].values[0]
        
        if len(starter_career[starter_career.starter == today_games.iloc[k]["road_starter"]]) == 0:
            
            road_starter_career = np.zeros(len(starter_career_metrics))
        
        else:
            
            road_starter_career = starter_career[starter_career.starter == \
                                            today_games.iloc[k]["road_starter"]][starter_career_metrics].values[0]
        
        if len(starter_season[starter_season.starter == today_games.iloc[k]["home_starter"]]) == 0:
            
            home_starter_season = np.zeros(len(starter_season_metrics))
            
        else:
        
            home_starter_season = starter_season[starter_season.starter == \
                                            today_games.iloc[k]["home_starter"]][starter_season_metrics].values[0]
        
        if len(starter_career[starter_career.starter == today_games.iloc[k]["home_starter"]]) == 0:
            
            home_starter_career = np.zeros(len(starter_career_metrics))
            
        else:
        
            home_starter_career = starter_career[starter_career.starter == \
                                            today_games.iloc[k]["home_starter"]][starter_career_metrics].values[0]
        
        for x, y, z in zip(team_metrics, road_stats, home_stats):
            
            di["road" + x] = y
            
            di["home" + x] = z
        
        for a, b, c, d, e, f in zip(starter_season_metrics, starter_career_metrics,
                          road_starter_season, road_starter_career,
                          home_starter_season, home_starter_career):
            
            di["road" + a] = c
            
            di["road" + b] = d
            
            di["home" + a] = e
            
            di["home" + b] = f
            
        all_inputs.append(di)
        
    return(all_inputs)

def get_momentum_model_inputs(today, past):
    
    all_momentum = []
    
    for k in range(len(today)):
    
        home_team = today.iloc[k]["home_team"]

        road_team = today.iloc[k]["road_team"]

        d = {}

        keep_cols = ["date", "is_doubleheader", "is_tripleheader", "home_team", "road_team"]

        for col in keep_cols:

            d[col] = today.iloc[k][col]

        home_full = past[(past.home_team == home_team) | (past.road_team == home_team)].reset_index(drop = True)

        road_full = past[(past.home_team == road_team) | (past.road_team == road_team)].reset_index(drop = True)

        home_df = past[past.home_team == home_team].reset_index(drop = True)

        road_df = past[past.road_team == road_team].reset_index(drop = True)

        if len(home_df) == 0:

            d["home_record"] = 0

            d["run_differential_hm"] = 0

            d["avg_margin_hm"] = 0

            d["home_H_streak"] = 0

        else:

            d["home_record"] = len(home_df[home_df.score1 > home_df.score2]) - len(home_df[home_df.score2 > home_df.score1])

            d["run_differential_hm"] = home_df.score1.sum() - home_df.score2.sum()

            d["avg_margin_hm"] = d["run_differential_hm"] / len(home_df)

            d["home_H_streak"] = monitor_streak(home_team, home_df, style = "home")

        if len(home_full) == 0:

            d["home_streak"] = 0

        else:

            d["home_streak"] = monitor_streak(home_team, home_full, style = "overall")

        if len(road_df) == 0:

            d["road_record"] = 0

            d["run_differential_rd"] = 0

            d["avg_margin_rd"] = 0

            d["road_R_streak"] = 0

        else:

            d["road_record"] = len(road_df[road_df.score2 > road_df.score1]) - len(road_df[road_df.score1 > road_df.score2])

            d["run_differential_rd"] = road_df.score2.sum() - road_df.score1.sum()

            d["avg_margin_rd"] = d["run_differential_rd"] / len(road_df)

            d["road_R_streak"] = monitor_streak(road_team, road_df, style = "road")

        if len(road_full) == 0:

            d["road_streak"] = 0

        else:

            d["road_streak"] = monitor_streak(road_team, road_full, style = "overall")

        all_momentum.append(d)
        
    return(all_momentum)

def calculate_distance_model_inputs(working, past):
    
    distances = []
    
    for l in range(len(working)):
        
        road_team = working.iloc[l]["road_team"]
        
        road_full = past[(past.home_team == road_team) | (past.road_team == road_team)]
        
        current_opponent = None

        k = -1

        while True:

            if not current_opponent:

                current_opponent = road_full.iloc[k]["home_team"]

                total_distance = haversine_distance(road_full.iloc[k].home_latitude, 
                                               road_full.iloc[k].home_longitude,
                                               road_full.iloc[k].road_latitude,
                                               road_full.iloc[k].road_longitude)
                k -= 1

                next_opponent = road_full.iloc[k]["home_team"]

            elif current_opponent == next_opponent:

                k -= 1

                next_opponent = road_full.iloc[k]["home_team"]

            elif next_opponent == road_team:

                break

            else:

                total_distance += haversine_distance(road_full.iloc[k].home_latitude, 
                                               road_full.iloc[k].home_longitude,
                                               road_full.iloc[k].road_latitude,
                                               road_full.iloc[k].road_longitude)
                k -= 1

                next_opponent = road_full.iloc[k]["home_team"]
                
        distances.append(total_distance)
        
    return(distances)