'''
Utility function for model predictions / deployment
'''
import pandas as pd 

def id_converter(current_season, all_time):
    
    all_current = set(list(current_season.home_starter.unique()) + list(current_season.road_starter.unique()))

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

def create_lines(x):
    
    if x > 0.5:
        
        return((x / (1. - x)) * -100)
    
    else:
        
        return(((1. / x) - 1) * 100)

def generate_lines(model, today_games, model_inputs):
    
    home_preds =  model.predict(model_inputs)
    
    home_lines = pd.Series([i[0] for i in home_preds])
    
    home_lines = home_lines.apply(lambda x: create_lines(x)).rename("model_home").astype("int32")
    
    today_games = pd.concat([today_games, home_lines], axis = 1)
    
    return(today_games)






