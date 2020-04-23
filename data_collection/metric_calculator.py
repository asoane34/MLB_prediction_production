'''
Transformations applied to raw data- Used to generate model inputs for prediction, not used to generate training samples.
'''

from dataclasses import dataclass, field
import pandas as pd 
import numpy as np 
from utils import aggregate, aggregate_starter_career, aggregate_starter_season

wOBA = ["_BB", "_HBP", "_1B", "_2B", "_3B", "_HR"]

wOBA_denom = ["_AB", "_BB", "_SAC", "_HBP"]

OBP = ["_H", "_BB", "_IBB", "_HBP"]

OBP_denom = ["_AB", "_BB", "_HBP", "_SAC"]

SLG = ["_1B", "_2B", "_3B", "_HR"]

SLG_mlt = np.array([1, 2, 3, 4], dtype = "int32")

wOBA_R = ["_relief_BB", "_relief_HBP", "_relief_1B", "_relief_2B", "_relief_3B", "_relief_HR"]

wOBA_denom_R = ["_relief_AB", "_relief_BB", "_relief_SAC", "_relief_HBP"]

wOBA_S = ["_starter_BB", "_starter_HBP", "_starter_1B", "_starter_2B", "_starter_3B", "_starter_HR"]

wOBA_denom_S = ["_starter_AB", "_starter_BB", "_starter_SAC", "_starter_HBP"]

@dataclass
class Calculator():
    season_totals: pd.core.frame.DataFrame
    starter_season: pd.core.frame.DataFrame
    starter_career: pd.core.frame.DataFrame
    

    def create_metrics(self):

        self.season_totals["_wOBA"] = self.season_totals.apply(lambda x: self.calc_wOBA(x, wOBA, wOBA_denom), axis = 1)

        self.season_totals["_relief_wOBA"] = self.season_totals.apply(lambda x: self.calc_wOBA(x, wOBA_R, 
                                                                                wOBA_denom_R, style = "relief"), 
                                                        axis = 1)

        self.starter_career["_starter_career_wOBA"] = self.starter_career.apply(lambda x: self.calc_wOBA(x, wOBA_S,
                                                                wOBA_denom_S, style = "starter"),
                                            axis = 1)

        self.starter_season["_starter_season_wOBA"] = self.starter_season.apply(lambda x: self.calc_wOBA(x, wOBA_S,
                                                                wOBA_denom_S, style = "starter"),
                                      axis = 1)

        self.season_totals["_wRAA"] = self.season_totals.apply(lambda x: self.calc_wRAA(x), axis = 1)

        self.season_totals["_relief_wRAA"] = self.season_totals.apply(lambda x: self.calc_wRAA(x, style = "relief"), 
                                                        axis = 1)

        self.starter_career["_starter_career_wRAA"] = self.starter_career.apply(lambda x: self.calc_wRAA(x, style = "career"),
                                            axis = 1)

        self.starter_season["_starter_season_wRAA"] = self.starter_season.apply(lambda x: self.calc_wRAA(x, style = "season"),
                                            axis = 1)

        self.season_totals["_wRC"] = self.season_totals.apply(lambda x: self.calc_wRC(x), axis = 1)

        self.season_totals["_relief_wRC"] = self.season_totals.apply(lambda x: self.calc_wRC(x, style = "relief"), 
                                                        axis = 1)

        self.starter_career["_starter_career_wRC"] = self.starter_career.apply(lambda x: self.calc_wRC(x, style = "career"),
                                            axis = 1)

        self.starter_season["_starter_season_wRC"] = self.starter_season.apply(lambda x: self.calc_wRC(x, style = "season"),
                                            axis = 1)

        self.season_totals["_OPS"] = self.season_totals.apply(lambda x: self.calc_OPS(x, OBP, OBP_denom, SLG, SLG_mlt), axis = 1)

        self.season_totals["_relief_FIP"] = self.season_totals.apply(lambda x: self.calc_FIP(x), axis = 1)

        self.starter_career["_starter_career_FIP"] = self.starter_career.apply(lambda x: self.calc_FIP(x, style = "starter"), axis = 1)

        self.starter_season["_starter_season_FIP"] = self.starter_season.apply(lambda x: self.calc_FIP(x, style = "starter"), axis = 1)

        self.season_totals["_relief_WHIP"] = self.season_totals.apply(lambda x: self.calc_WHIP(x), axis = 1)

        self.starter_career["_starter_career_WHIP"] = self.starter_career.apply(lambda x: self.calc_WHIP(x, style = "starter"), axis = 1)

        self.starter_season["_starter_season_WHIP"] = self.starter_season.apply(lambda x: self.calc_WHIP(x, style = "starter"), axis = 1)

        self.season_totals["_relief_ERA"] = self.season_totals.apply(lambda x: self.calc_ERA(x), axis = 1)

        self.starter_career["_starter_career_ERA"] = self.starter_career.apply(lambda x: self.calc_ERA(x, style = "starter"), axis = 1)

        self.starter_season["_starter_season_ERA"] = self.starter_season.apply(lambda x: self.calc_ERA(x, style = "starter"), axis = 1)

        self.season_totals["_relief_K_9"] = self.season_totals.apply(lambda x: self.calc_K9(x), axis = 1)

        self.starter_career["_starter_careerK/9"] = self.starter_career.apply(lambda x: self.calc_K9(x, style = "starter"), axis = 1)

        self.starter_season["_starter_seasonK/9"] = self.starter_season.apply(lambda x: self.calc_K9(x, style = "starter"), axis = 1)

        self.season_totals["_relief_K_BB"] = self.season_totals.apply(lambda x: self.calc_KBB(x), axis = 1)

        self.starter_career["_starter_careerK/BB"] = self.starter_career.apply(lambda x: self.calc_KBB(x, style = "starter"), axis = 1)

        self.starter_season["_starter_seasonK/BB"] = self.starter_season.apply(lambda x: self.calc_KBB(x, style = "starter"), axis = 1)
        
        self.starter_career["_starter_career_AVGIP"] = self.starter_career.apply(lambda x: self.calc_AVGIP(x), axis = 1)
        
        self.starter_season["_starter_seasonAVG_IP"] = self.starter_season.apply(lambda x: self.calc_AVGIP(x), axis = 1)

        self.season_totals.to_csv("./all_data/season_totals.csv", index = False)

        self.starter_season.to_csv("./all_data/starter_season.csv", index = False)

        self.starter_career.to_csv("./all_data/starter_career.csv", index = False)

        print("All files written")
            
    @staticmethod
    def calc_wOBA(obs, num, denom, style = "offense"):

        if style == "offense":

            return(

                sum(obs[num].values * obs[["wBB", "wHBP", "w1B", "w2B", "w3B", "wHR"]]) /\
                     (sum(obs[denom].values) - obs["_IBB"])
            )

        elif style == "relief":

            return(

                sum(obs[num].values * obs[["wBB", "wHBP", "w1B", "w2B", "w3B", "wHR"]]) /\
                     (sum(obs[denom].values) - obs["_relief_IBB"])
            )

        else:

            return(

                sum(obs[num].values * obs[["wBB", "wHBP", "w1B", "w2B", "w3B", "wHR"]].values) /\
                    (sum(obs[denom].values) - obs["_starter_IBB"])
            )

    @staticmethod
    def calc_wRAA(obs, style = "offense"):
    
        if style == "offense":

            return(

                ((obs._wOBA - obs.wOBA) / obs.wOBAScale) * (obs._PA)
            )

        elif style == "relief":

            return(

                ((obs._relief_wOBA - obs.wOBA) / obs.wOBAScale) * (obs._relief_PA)
            )
            
        elif style == "season":

            return(

                ((obs._starter_season_wOBA - obs.wOBA_constant) / obs.wOBA_scale) * (obs._starter_PA)
            )
            
        else:
            
            return(

                    ((obs._starter_career_wOBA - obs.wOBA_constant) / obs.wOBA_scale) * (obs._starter_PA)
                )

    @staticmethod
    def calc_wRC(obs, style = "offense"):
    
        if style == "offense":
        
            return(
            
                (((obs._wRAA / obs._PA) + obs["R/PA"]) + (obs["R/PA"] - (obs.batting_park_factor * obs["R/PA"]))) / 
                ((obs.league_wRC / obs.league_PA)) * 100
            )

        elif style == "relief":

            return(
            
                (((obs._relief_wRAA / obs._relief_PA) + obs["R/PA"]) + (obs["R/PA"] -\
                                                                        (obs.batting_park_factor * obs["R/PA"]))) / 
                ((obs.league_wRC / obs.league_PA)) * 100
            )
        
        elif style == "season":
            
            return(
            
                (((obs._starter_season_wRAA / obs._starter_PA) + obs.RPA) + (obs.RPA -\
                                                                        (obs.batting_park_factor * obs.RPA))) / 
                ((obs.league_wRC / obs.league_PA)) * 100
            )
        
        else:
            
            return(
            
                (((obs._starter_career_wRAA / obs._starter_PA) + obs.RPA) + (obs.RPA -\
                                                                        (obs.batting_park_factor * obs.RPA))) / 
                ((obs.league_wRC / obs.league_PA)) * 100
            )

    @staticmethod
    def calc_OPS(obs, OBP, OBP_denom,
            SLG, SLG_mlt):
    
        OBP = sum(obs[OBP].values) / sum(obs[OBP_denom].values)
        
        SLG = sum(obs[SLG].values * SLG_mlt) / obs["_AB"]
        
        return(OBP + SLG)

    
    @staticmethod
    def calc_FIP(obs, style = "relief"):
    
        if style == "relief":

            return(

                (((13 * obs["_relief_HR"]) + (2 * sum(obs[["_relief_BB", 
                                                    "_relief_HBP", 
                                                    "_relief_IBB"]].values)) - (2 * obs["_relief_K"])) /
                obs["_relief_IP"]) + obs.cFIP
            )
        
        elif style == "starter":
            
            return(

                (((13 * obs["_starter_HR"]) + (2 * sum(obs[["_starter_BB", 
                                                    "_starter_HBP", 
                                                    "_starter_IBB"]].values)) - (2 * obs["_starter_K"])) /
                obs["_starter_IP"]) + obs.FIP_constant
            )

    @staticmethod
    def calc_WHIP(obs, style = "relief"):
    
        if style == "relief":
        
            return(
            
                sum(obs[["_relief_H", "_relief_BB", "_relief_IBB"]].values) / obs["_relief_IP"]
            )
        
        elif style == "starter":
            
            return(
            
                sum(obs[["_starter_H", "_starter_BB", "_starter_IBB"]].values) / obs["_starter_IP"]
            )

    @staticmethod
    def calc_ERA(obs, style = "relief"):

        if style == "relief":
        
            return(
            
                (obs["_relief_ER"] / obs["_relief_IP"]) * 9
            )
        
        elif style == "starter":
            
            return(
            
                (obs["_starter_ER"] / obs["_starter_IP"]) * 9
            )
        
    @staticmethod
    def calc_K9(obs, style = "relief"):

        if style == "relief":
        
            return(
            
                (obs["_relief_K"] / obs["_relief_IP"]) * 9
            )
        
        elif style == "starter":
            
            return(
            
                (obs["_starter_K"] / obs["_starter_IP"]) * 9
            )
        
    @staticmethod
    def calc_KBB(obs, style = "relief"):

        if style == "relief":

            if obs._relief_BB == 0.:

                return(obs._relief_K)

            else:
        
                return(
                
                    obs["_relief_K"] / obs["_relief_BB"]
                )
        
        elif style == "starter":

            if obs._starter_BB == 0.:

                return(obs._starter_K)

            else:
            
                return(
                
                    obs["_starter_K"] / obs["_starter_BB"]
                )
    @staticmethod
    def calc_AVGIP(obs):
        
        return(obs._starter_IP / obs.n_starts)

if __name__ == "__main__":

    print("Beginning current season stat aggregation...")
    
    current_season = pd.read_csv("./all_data/current_season.csv")

    season_totals = aggregate(current_season)

    print("Beginning starting pitcher stat aggregation...")

    career_data = pd.read_csv("./all_data/past_raw.csv.gz", compression = "gzip")

    career_starter = aggregate_starter_career(current_season, career_data)

    season_starter = aggregate_starter_season(current_season, career_data)

    print("Beginning metric calculation...")

    calc = Calculator(season_totals, season_starter, career_starter)

    calc.create_metrics()

    print("All files written successfully.")



