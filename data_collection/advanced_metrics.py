from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
import pandas as pd 
import numpy as np 


@dataclass
class AdvancedMetricsCreator():
    master: pd.core.frame.DataFrame
    n_jobs: int = None
        
    def recreate(self):
        
        all_results = []
        
        with ProcessPoolExecutor(max_workers = self.n_jobs) as executor:
            
            for k in self.master.index:
                
                if k > 0 and k % 10000 == 0:
                    
                    print("{} observations processed".format(str(k)))
                        
                prior = self.master.iloc[:k+1]
                
                all_results.append(executor.submit(self.calc_stats, prior, k))
                
        all_results = [i.result() for i in all_results]
        
        return(all_results)
              
    @staticmethod
    def calc_stats(prior_df: pd.core.frame.DataFrame, k: int):
        
        game_master= {}
        
        game_master["date"] = prior_df.iloc[k]["date"]
        
        game_master["is_doubleheader"] = prior_df.iloc[k]["is_doubleheader"]
        
        game_master["is_tripleheader"] = prior_df.iloc[k]["is_tripleheader"]
        
        season = prior_df.iloc[k]["season"]
        
        home_team = prior_df.iloc[k]["home_team"]
        
        road_team = prior_df.iloc[k]["road_team"]
        
        home_starter = prior_df.iloc[k]["home_starter"]
        
        road_starter = prior_df.iloc[k]["road_starter"]
        
        game_master["season"] = season
        
        game_master["home_team"] = home_team
        
        game_master["road_team"] = road_team
        
        game_master["home_starter"] = home_starter
        
        game_master["road_starter"] = road_starter
        
        wOBA_weights = ["wBB", "wHBP", "w1B", "w2B", "w3B", "wHR"]

        wOBA_home = ["home_BB", "home_HBP", "home_1B", "home_2B", "home_3B", "home_HR"]

        wOBA_road = ["road_BB", "road_HBP", "road_1B", "road_2B", "road_3B", "road_HR"]

        denom_home = ["home_AB", "home_BB", "home_SAC", "home_HBP"]

        denom_road = ["road_AB", "road_BB", "road_SAC", "road_HBP"]

        home_IBB = "home_IBB"

        road_IBB = "road_IBB"
        
        wOBA_home_SP = ["home_starter_BB", "home_starter_HBP", "home_starter_1B", "home_starter_2B", 
                 "home_starter_3B", "home_starter_HR"]

        wOBA_road_SP = ["road_starter_BB", "road_starter_HBP", "road_starter_1B", "road_starter_2B", 
                     "road_starter_3B", "road_starter_HR"]

        wOBA_home_R = ["home_relief_BB", "home_relief_HBP", "home_relief_1B", "home_relief_2B", 
                     "home_relief_3B", "home_relief_HR"]

        wOBA_road_R = ["road_relief_BB", "road_relief_HBP", "road_relief_1B", "road_relief_2B", 
                     "road_relief_3B", "road_relief_HR"]

        denom_home_SP = ["home_starter_AB", "home_starter_BB", "home_starter_SAC", "home_starter_HBP"]

        denom_road_SP = ["road_starter_AB", "road_starter_BB", "road_starter_SAC", "road_starter_HBP"]

        denom_home_R = ["home_relief_AB", "home_relief_BB", "home_relief_SAC", "home_relief_HBP"]

        denom_road_R = ["road_relief_AB", "road_relief_BB", "road_relief_SAC", "road_relief_HBP"]

        home_IBB_SP = "home_starter_IBB"

        road_IBB_SP = "road_starter_IBB"

        home_IBB_R = "home_relief_IBB"

        road_IBB_R = "road_relief_IBB"
        
        OBP_home = ["home_H", "home_BB", "home_IBB", "home_HBP"]
        
        OBP_denomH = ["home_AB", "home_BB", "home_HBP", "home_SAC"]
        
        OBP_road = ["road_H", "road_BB", "road_IBB", "road_HBP"]
        
        OBP_denomR = ["road_AB", "road_BB", "road_HBP", "road_SAC"]
        
        SLG_home = ["home_1B", "home_2B", "home_3B", "home_HR"]
        
        SLG_road = ["road_1B", "road_2B", "road_3B", "road_HR"]
        
        SLG_mlt = np.array([1, 2, 3, 4])
        
        prefixes = ["home_", "road_"]
        
        for prefix in prefixes:
            
            if prefix == "home_":
                
                home_df = prior_df[(prior_df.home_team == home_team) & (prior_df.season == season)]
                
                road_df = prior_df[(prior_df.road_team == home_team) & (prior_df.season == season)]
                
                home_s_df = home_df[home_df.home_starter == home_starter]
                
                road_s_df = road_df[road_df.road_starter == home_starter]
                
                home_career = prior_df[prior_df.home_starter == home_starter]
                
                road_career = prior_df[prior_df.road_starter == home_starter]
                
            else:
                
                home_df = prior_df[(prior_df.home_team == road_team) & (prior_df.season == season)]
                
                road_df = prior_df[(prior_df.road_team == road_team) & (prior_df.season == season)]
                
                home_s_df = home_df[home_df.home_starter == road_starter]
                
                road_s_df = road_df[road_df.road_starter == road_starter]
                
                home_career = prior_df[prior_df.home_starter == road_starter]
                
                road_career = prior_df[prior_df.road_starter == road_starter]
        
            if len(home_df) != 0 or len(road_df) != 0:

                if len(home_df) != 0:
                    
                    park_factor = pd.concat([home_df.home_batting_park_factor, 
                                            road_df.home_batting_park_factor], axis = 0).mean()

                    wOBA = sum((home_df[wOBA_home].sum().values + road_df[wOBA_road].sum().values) * \
                                home_df[wOBA_weights].max().values) /\
                                (sum(home_df[denom_home].sum().values + road_df[denom_road].sum().values) -\
                                home_df[home_IBB].sum() - road_df[road_IBB].sum())

                    wRAA = ((wOBA - home_df.wOBA.max()) / home_df.wOBAScale.max()) * \
                            (home_df.home_PA.sum() + road_df.road_PA.sum())

                    wRC = ((((wRAA / (home_df.home_PA.sum() + road_df.road_PA.sum())) + home_df["R/PA"].max()) +\
                            (home_df["R/PA"].max() - (park_factor * home_df["R/PA"].max()))) /\
                            (home_df.home_league_wRC.max() / home_df.home_league_PA.max())) * 100
                    
                    if home_df.home_relief_IP.sum() + road_df.road_relief_IP.sum() == 0.:
                        
                        relief_wOBA = 0.
                        
                        relief_wRAA = 0.
                        
                        relief_wRC = 0.
                        
                    else:
                        
                        relief_wOBA = sum((home_df[wOBA_home_R].sum().values + road_df[wOBA_road_R].sum().values) * \
                                    home_df[wOBA_weights].max().values) /\
                                    (sum(home_df[denom_home_R].sum().values + road_df[denom_road_R].sum().values) -\
                                    home_df[home_IBB_R].sum() - road_df[road_IBB_R].sum())

                        relief_wRAA = ((relief_wOBA - home_df.wOBA.max()) / home_df.wOBAScale.max()) * \
                                      (home_df.home_relief_PA.sum() + road_df.road_relief_PA.sum())

                        relief_wRC = ((((relief_wRAA / (home_df.home_relief_PA.sum() +\
                                                        road_df.road_relief_PA.sum())) + home_df["R/PA"].max()) +\
                                    (home_df["R/PA"].max() - (park_factor * home_df["R/PA"].max()))) /\
                                    (home_df.home_league_wRC.max() / home_df.home_league_PA.max())) * 100
                    
                    if home_df.home_relief_IP.sum() + road_df.road_relief_IP.sum() == 0.:
                        
                        FIP = 0.
                        
                    else:
                    
                        FIP = (((13 * (home_df.home_relief_HR.sum() + road_df.road_relief_HR.sum())) +\
                              (3 * (home_df.home_relief_BB.sum() + home_df.home_relief_HBP.sum() +\
                                   home_df.home_relief_IBB.sum() + road_df.road_relief_BB.sum() +\
                                   road_df.road_relief_HBP.sum() + road_df.road_relief_IBB.sum())) -\
                              (2 * (home_df.home_relief_K.sum() + road_df.road_relief_K.sum()))) /\
                              (home_df.home_relief_IP.sum() + road_df.road_relief_IP.sum())) + \
                              home_df.cFIP.max()

                else:
                    
                    park_factor = road_df.home_batting_park_factor.mean()

                    wOBA = sum((home_df[wOBA_home].sum().values + road_df[wOBA_road].sum().values) * \
                                road_df[wOBA_weights].max().values) /\
                                (sum(home_df[denom_home].sum().values + road_df[denom_road].sum().values) -\
                                home_df[home_IBB].sum() - road_df[road_IBB].sum())

                    wRAA = ((wOBA - road_df.wOBA.max()) / road_df.wOBAScale.max()) * \
                            (home_df.home_PA.sum() + road_df.road_PA.sum())

                    wRC = ((((wRAA / (home_df.home_PA.sum() + road_df.road_PA.sum())) + road_df["R/PA"].max()) +\
                            (road_df["R/PA"].max() - (park_factor * road_df["R/PA"].max()))) /\
                            (road_df.road_league_wRC.max() / road_df.road_league_PA.max())) * 100
                    
                    if road_df.road_relief_IP.sum() == 0.:
                        
                        relief_wOBA = 0.
                        
                        relief_wRAA = 0.
                        
                        relief_wRC = 0.
                        
                    else:
                    
                        relief_wOBA = sum((home_df[wOBA_home_R].sum().values + road_df[wOBA_road_R].sum().values) * \
                                    road_df[wOBA_weights].max().values) /\
                                    (sum(home_df[denom_home_R].sum().values + road_df[denom_road_R].sum().values) -\
                                    home_df[home_IBB_R].sum() - road_df[road_IBB_R].sum())

                        relief_wRAA = ((relief_wOBA - road_df.wOBA.max()) / road_df.wOBAScale.max()) * \
                                      (home_df.home_relief_PA.sum() + road_df.road_relief_PA.sum())

                        relief_wRC = ((((relief_wRAA / (home_df.home_relief_PA.sum() +\
                                                        road_df.road_relief_PA.sum())) + road_df["R/PA"].max()) +\
                                    (road_df["R/PA"].max() - (park_factor * road_df["R/PA"].max()))) /\
                                    (road_df.road_league_wRC.max() / road_df.road_league_PA.max())) * 100
                    
                    if road_df.road_relief_IP.sum() == 0.:
                        
                        FIP = 0.
                        
                    else:
                    
                        FIP = (((13 * (home_df.home_relief_HR.sum() + road_df.road_relief_HR.sum())) +\
                              (3 * (home_df.home_relief_BB.sum() + home_df.home_relief_HBP.sum() +\
                                   home_df.home_relief_IBB.sum() + road_df.road_relief_BB.sum() +\
                                   road_df.road_relief_HBP.sum() + road_df.road_relief_IBB.sum())) -\
                              (2 * (home_df.home_relief_K.sum() + road_df.road_relief_K.sum()))) /\
                              (home_df.home_relief_IP.sum() + road_df.road_relief_IP.sum())) + \
                              road_df.cFIP.max()

                OBP = sum(home_df[OBP_home].sum().values + road_df[OBP_road].sum().values) /\
                      sum(home_df[OBP_denomH].sum().values + road_df[OBP_denomR].sum().values)

                SLG = (sum(home_df[SLG_home].sum().values * SLG_mlt) +\
                       sum(road_df[SLG_road].sum().values * SLG_mlt)) / (home_df["home_AB"].sum() +\
                                                                        road_df["road_AB"].sum())
                
                if home_df.home_relief_IP.sum() + road_df.road_relief_IP.sum() == 0.:
                    
                    WHIP = 0.
                    
                    ERA = 0.
                    
                    K_9 = 0.
                    
                else:
                
                    WHIP = (home_df.home_relief_H.sum() + home_df.home_relief_BB.sum() +\
                           home_df.home_relief_IBB.sum() + road_df.road_relief_H.sum() +\
                           road_df.road_relief_BB.sum() + road_df.road_relief_IBB.sum()) /\
                           (home_df.home_relief_IP.sum() + road_df.road_relief_IP.sum())
                
                    ERA = ((home_df.home_relief_ER.sum() + road_df.road_relief_ER.sum()) /\
                          (home_df.home_relief_IP.sum() + road_df.road_relief_IP.sum())) * 9
                    
                    K_9 = ((home_df.home_relief_K.sum() + road_df.road_relief_K.sum()) /\
                          (home_df.home_relief_IP.sum() + road_df.road_relief_IP.sum())) * 9
                
                if home_df.home_relief_BB.sum() + road_df.road_relief_BB.sum() == 0:
                    
                    K_BB = home_df.home_relief_K.sum() + road_df.road_relief_K.sum()
                    
                else:
                
                    K_BB = (home_df.home_relief_K.sum() + road_df.road_relief_K.sum()) /\
                           (home_df.home_relief_BB.sum() + road_df.road_relief_BB.sum())
                
                game_master[prefix + "wOBA"] = wOBA

                game_master[prefix + "wRAA"] = wRAA

                game_master[prefix + "wRC"] = wRC

                game_master[prefix + "OPS"] = OBP + SLG
                
                game_master[prefix + "relief_wOBA"] = relief_wOBA
                
                game_master[prefix + "relief_wRAA"] = relief_wRAA
                
                game_master[prefix + "relief_wRC"] = relief_wRC
                
                game_master[prefix + "relief_FIP"] = FIP
                
                game_master[prefix + "relief_WHIP"] = WHIP
                
                game_master[prefix + "relief_ERA"] = ERA
                
                game_master[prefix + "relief_K_BB"] = K_BB
                
                game_master[prefix + "relief_K_9"] = K_9

            else:

                game_master[prefix + "wOBA"] = 0.

                game_master[prefix + "wRAA"] = 0.

                game_master[prefix + "wRC"] = 0.

                game_master[prefix + "OPS"] = 0.
                
                game_master[prefix + "relief_wOBA"] = 0.
                
                game_master[prefix + "relief_wRAA"] = 0.
                
                game_master[prefix + "relief_wRC"] = 0. 
                
                game_master[prefix + "relief_FIP"] = 0.
                
                game_master[prefix + "relief_WHIP"] = 0.
                
                game_master[prefix + "relief_ERA"] = 0.
                
                game_master[prefix + "relief_K_BB"] = 0.
                
                game_master[prefix + "relief_K_9"] = 0.
                
            if len(home_career) != 0 or len(road_career) != 0:
                
                if len(home_career) != 0:
                    
                    cpark_factor = pd.concat([home_career.home_batting_park_factor, 
                                            road_career.home_batting_park_factor], axis = 0).mean()
                    
                    cwOBA_avg = pd.concat([home_career.wOBA, road_career.wOBA], axis = 0).mean()
                    
                    cwOBA_scale = pd.concat([home_career.wOBAScale, road_career.wOBAScale],
                                            axis = 0).mean()
                    
                    cRPA = pd.concat([home_career["R/PA"], road_career["R/PA"]], axis = 0).mean()
                    
                    cleague_wRC = pd.concat([home_career.home_league_wRC, road_career.road_league_wRC],
                                           axis = 0).mean()
                    
                    cleague_PA = pd.concat([home_career.home_league_PA, road_career.road_league_PA],
                                          axis = 0).mean()
                    
                    cwOBA = sum((home_career[wOBA_home_SP].sum().values + road_career[wOBA_road_SP].sum().values) * \
                                home_career[wOBA_weights].max().values) /\
                            (sum(home_career[denom_home_SP].sum().values +road_career[denom_road_SP].sum().values)\
                                 - home_career[home_IBB_SP].sum() - road_career[road_IBB_SP].sum())

                    cwRAA = ((cwOBA - cwOBA_avg) / cwOBA_scale) * \
                            (home_career.home_starter_PA.sum() + road_career.road_starter_PA.sum())

                    cwRC = ((((cwRAA / (home_career.home_starter_PA.sum() +\
                                        road_career.road_starter_PA.sum())) + cRPA) +\
                            (cRPA - (cpark_factor * cRPA))) /\
                            (cleague_wRC / cleague_PA)) * 100
                    
                    if home_career.home_starter_IP.sum() + road_career.road_starter_IP.sum() == 0.:
                        
                        caFIP = "inf"
                        
                    else:
                    
                        caFIP = (((13 * (home_career.home_starter_HR.sum() + road_career.road_starter_HR.sum())) +\
                              (3 * (home_career.home_starter_BB.sum() + home_career.home_starter_HBP.sum() +\
                                   home_career.home_starter_IBB.sum() + road_career.road_starter_BB.sum() +\
                                   road_career.road_starter_HBP.sum() + road_career.road_starter_IBB.sum())) -\
                              (2 * (home_career.home_starter_K.sum() + road_career.road_starter_K.sum()))) /\
                              (home_career.home_starter_IP.sum() + road_career.road_starter_IP.sum())) + \
                              home_career.cFIP.mean()
    
                else:
            
                    cpark_factor = road_career.home_batting_park_factor.mean()
                    
                    cwOBA_avg = road_career.wOBA.mean()
                    
                    cwOBA_scale = road_career.wOBAScale.mean()
                    
                    cRPA = road_career["R/PA"].mean()
                    
                    cleague_wRC = road_career.road_league_wRC.mean()
                    
                    cleague_PA = road_career.road_league_PA.mean()
                    
                    cwOBA = sum((home_career[wOBA_home_SP].sum().values + road_career[wOBA_road_SP].sum().values) * \
                                road_career[wOBA_weights].max().values) /\
                            (sum(home_career[denom_home_SP].sum().values +road_career[denom_road_SP].sum().values)\
                                 - home_career[home_IBB_SP].sum() - road_career[road_IBB_SP].sum())

                    cwRAA = ((cwOBA - cwOBA_avg) / cwOBA_scale) * \
                            (home_career.home_starter_PA.sum() + road_career.road_starter_PA.sum())

                    cwRC = ((((cwRAA / (home_career.home_starter_PA.sum() +\
                                        road_career.road_starter_PA.sum())) + cRPA) +\
                            (cRPA - (cpark_factor * cRPA))) /\
                            (cleague_wRC / cleague_PA)) * 100
        
                    if road_career.road_starter_IP.sum() == 0.:
                        
                        caFIP = "inf"
                    
                    else:
                    
                        caFIP = (((13 * (home_career.home_starter_HR.sum() + road_career.road_starter_HR.sum())) +\
                              (3 * (home_career.home_starter_BB.sum() + home_career.home_starter_HBP.sum() +\
                                   home_career.home_starter_IBB.sum() + road_career.road_starter_BB.sum() +\
                                   road_career.road_starter_HBP.sum() + road_career.road_starter_IBB.sum())) -\
                              (2 * (home_career.home_starter_K.sum() + road_career.road_starter_K.sum()))) /\
                              (home_career.home_starter_IP.sum() + road_career.road_starter_IP.sum())) + \
                              road_career.cFIP.mean()
                
                if home_career.home_starter_IP.sum() + road_career.road_starter_IP.sum() == 0.:
                    
                    caWHIP = "inf"
                    
                    caERA = "inf"
                    
                    caK_9 = 0.
                    
                    caAVG_IP = 0.
                    
                else:
                    
                    caWHIP = (home_career.home_starter_H.sum() + home_career.home_starter_BB.sum() +\
                           home_career.home_starter_IBB.sum() + road_career.road_starter_H.sum() +\
                           road_career.road_starter_BB.sum() + road_career.road_starter_IBB.sum()) /\
                           (home_career.home_starter_IP.sum() + road_career.road_starter_IP.sum())

                    caERA = ((home_career.home_starter_ER.sum() + road_career.road_starter_ER.sum()) /\
                          (home_career.home_starter_IP.sum() + road_career.road_starter_IP.sum())) * 9
                    
                    caK_9 = ((home_career.home_starter_K.sum() + road_career.road_starter_K.sum()) /\
                          (home_career.home_starter_IP.sum() + road_career.road_starter_IP.sum())) * 9
                    
                    caAVG_IP = (home_career.home_starter_IP.sum() + road_career.road_starter_IP.sum()) /\
                                (len(home_career) + len(road_career))
                
                if home_career.home_starter_BB.sum() + road_career.road_starter_BB.sum() == 0:
                    
                    caK_BB = home_career.home_starter_K.sum() + road_career.road_starter_K.sum()
                    
                else:
                    caK_BB = (home_career.home_starter_K.sum() + road_career.road_starter_K.sum()) /\
                             (home_career.home_starter_BB.sum() + road_career.road_starter_BB.sum())
                 
                game_master[prefix + "starter_career_wOBA"] = cwOBA
                
                game_master[prefix + "starter_career_wRAA"] = cwRAA
                
                game_master[prefix + "starter_career_wRC"] = cwRC
                
                game_master[prefix + "starter_career_FIP"] = caFIP
                
                game_master[prefix + "starter_career_WHIP"] = caWHIP
                
                game_master[prefix + "starter_career_ERA"] = caERA
                
                game_master[prefix + "starter_careerK/BB"] = caK_BB
                
                game_master[prefix + "starter_careerK/9"] = caK_9
                
                game_master[prefix + "starter_career_AVGIP"] = caAVG_IP
                
            else:
                
                game_master[prefix + "starter_career_wOBA"] = 0.
                
                game_master[prefix + "starter_career_wRAA"] = 0.
                
                game_master[prefix + "starter_career_wRC"] = 0.
                
                game_master[prefix + "starter_career_FIP"] = 0.
                
                game_master[prefix + "starter_career_WHIP"] = 0.
                
                game_master[prefix + "starter_career_ERA"] = 0.
                
                game_master[prefix + "starter_careerK/BB"] = 0.
                
                game_master[prefix + "starter_careerK/9"] = 0.
                
                game_master[prefix + "starter_career_AVGIP"] = 0.
                
            if len(home_s_df) != 0 or len(road_s_df) != 0:
                
                if len(home_s_df) != 0:
                    
                    spark_factor = pd.concat([home_s_df.home_batting_park_factor, 
                                            road_s_df.home_batting_park_factor], axis = 0).mean()
                    
                    swOBA = sum((home_s_df[wOBA_home_SP].sum().values + road_s_df[wOBA_road_SP].sum().values) * \
                                home_s_df[wOBA_weights].max().values) /\
                            (sum(home_s_df[denom_home_SP].sum().values +road_s_df[denom_road_SP].sum().values)\
                                 - home_s_df[home_IBB_SP].sum() - road_s_df[road_IBB_SP].sum())

                    swRAA = ((swOBA - home_s_df.wOBA.max()) / home_s_df.wOBAScale.max()) * \
                            (home_s_df.home_starter_PA.sum() + road_s_df.road_starter_PA.sum())

                    swRC = ((((swRAA / (home_s_df.home_starter_PA.sum() +\
                                        road_s_df.road_starter_PA.sum())) + home_s_df["R/PA"].max()) +\
                            (home_s_df["R/PA"].max() - (spark_factor * home_s_df["R/PA"].max()))) /\
                            (home_s_df.home_league_wRC.max() / home_s_df.home_league_PA.max())) * 100
                    
                    if home_s_df.home_starter_IP.sum() + road_s_df.road_starter_IP.sum() == 0.:
                        
                        sFIP = "inf"
                        
                    else:
                    
                        sFIP = (((13 * (home_s_df.home_starter_HR.sum() + road_s_df.road_starter_HR.sum())) +\
                              (3 * (home_s_df.home_starter_BB.sum() + home_s_df.home_starter_HBP.sum() +\
                                   home_s_df.home_starter_IBB.sum() + road_s_df.road_starter_BB.sum() +\
                                   road_s_df.road_starter_HBP.sum() + road_s_df.road_starter_IBB.sum())) -\
                              (2 * (home_s_df.home_starter_K.sum() + road_s_df.road_starter_K.sum()))) /\
                              (home_s_df.home_starter_IP.sum() + road_s_df.road_starter_IP.sum())) + \
                              home_s_df.cFIP.max()
                
                else:
                    
                    spark_factor = road_s_df.home_batting_park_factor.mean()
                    
                    swOBA = sum((home_s_df[wOBA_home_SP].sum().values + road_s_df[wOBA_road_SP].sum().values) * \
                                road_s_df[wOBA_weights].max().values) /\
                            (sum(home_s_df[denom_home_SP].sum().values +road_s_df[denom_road_SP].sum().values)\
                                 - home_s_df[home_IBB_SP].sum() - road_s_df[road_IBB_SP].sum())

                    swRAA = ((swOBA - road_s_df.wOBA.max()) / road_s_df.wOBAScale.max()) * \
                            (home_s_df.home_starter_PA.sum() + road_s_df.road_starter_PA.sum())

                    swRC = ((((swRAA / (home_s_df.home_starter_PA.sum() +\
                                        road_s_df.road_starter_PA.sum())) + road_s_df["R/PA"].max()) +\
                            (road_s_df["R/PA"].max() - (spark_factor * road_s_df["R/PA"].max()))) /\
                            (road_s_df.road_league_wRC.max() / road_s_df.road_league_PA.max())) * 100
                    
                    if road_s_df.road_starter_IP.sum() == 0.:
                        
                        sFIP = "inf"
                        
                    else:
                    
                        sFIP = (((13 * (home_s_df.home_starter_HR.sum() + road_s_df.road_starter_HR.sum())) +\
                              (3 * (home_s_df.home_starter_BB.sum() + home_s_df.home_starter_HBP.sum() +\
                                   home_s_df.home_starter_IBB.sum() + road_s_df.road_starter_BB.sum() +\
                                   road_s_df.road_starter_HBP.sum() + road_s_df.road_starter_IBB.sum())) -\
                              (2 * (home_s_df.home_starter_K.sum() + road_s_df.road_starter_K.sum()))) /\
                              (home_s_df.home_starter_IP.sum() + road_s_df.road_starter_IP.sum())) + \
                              road_s_df.cFIP.max()
                    
                if home_s_df.home_starter_IP.sum() + road_s_df.road_starter_IP.sum() == 0.:
                    
                    sWHIP = "inf"
                    
                    sERA = "inf"
                    
                    sK_9 = 0.
                    
                    sAVG_IP = 0.
                    
                else:
                
                    sWHIP = (home_s_df.home_starter_H.sum() + home_s_df.home_starter_BB.sum() +\
                           home_s_df.home_starter_IBB.sum() + road_s_df.road_starter_H.sum() +\
                           road_s_df.road_starter_BB.sum() + road_s_df.road_starter_IBB.sum()) /\
                           (home_s_df.home_starter_IP.sum() + road_s_df.road_starter_IP.sum())

                    sERA = ((home_s_df.home_starter_ER.sum() + road_s_df.road_starter_ER.sum()) /\
                          (home_s_df.home_starter_IP.sum() + road_s_df.road_starter_IP.sum())) * 9
                    
                    sK_9 = ((home_s_df.home_starter_K.sum() + road_s_df.road_starter_K.sum()) /\
                      (home_s_df.home_starter_IP.sum() + road_s_df.road_starter_IP.sum())) * 9
                    
                    sAVG_IP = (home_s_df.home_starter_IP.sum() + road_s_df.road_starter_IP.sum()) /\
                              (len(home_s_df) + len(road_s_df))
                
                if home_s_df.home_starter_BB.sum() + road_s_df.road_starter_BB.sum() == 0:
                    
                    sK_BB = home_s_df.home_starter_K.sum() + road_s_df.road_starter_K.sum()
                
                else:
                    sK_BB = (home_s_df.home_starter_K.sum() + road_s_df.road_starter_K.sum()) /\
                           (home_s_df.home_starter_BB.sum() + road_s_df.road_starter_BB.sum())
                
                game_master[prefix + "starter_season_wOBA"] = swOBA
                
                game_master[prefix + "starter_season_wRAA"] = swRAA
                
                game_master[prefix + "starter_season_wRC"] = swRC
                
                game_master[prefix + "starter_season_FIP"] = sFIP
                
                game_master[prefix + "starter_season_WHIP"] = sWHIP
                
                game_master[prefix + "starter_season_ERA"] = sERA
                
                game_master[prefix + "starter_seasonK/BB"] = sK_BB
                
                game_master[prefix + "starter_seasonK/9"] = sK_9
                
                game_master[prefix + "starter_seasonAVG_IP"] = sAVG_IP
                
            else:
                
                game_master[prefix + "starter_season_wOBA"] = 0.
                
                game_master[prefix + "starter_season_wRAA"] = 0.
                
                game_master[prefix + "starter_season_wRC"] = 0.
                
                game_master[prefix + "starter_season_FIP"] = 0.
                
                game_master[prefix + "starter_season_WHIP"] = 0.
                
                game_master[prefix + "starter_season_ERA"] = 0.
                
                game_master[prefix + "starter_seasonK/BB"] = 0.
                
                game_master[prefix + "starter_seasonK/9"] = 0.
                
                game_master[prefix + "starter_seasonAVG_IP"] = 0.
                        
        return(game_master)

if __name__ == "__main__":

    current_season = pd.read_csv("./all_data/current_season.csv")

    AMC = AdvancedMetricsCreator(current_season)

    master_list = AMC.recreate()

    compiled_df = pd.DataFrame(master_list)

    compiled_df.to_csv("./all_data/compiled_unstable.csv.gz", index = False, compression = "gzip")

    print("Game statistics compiled")

