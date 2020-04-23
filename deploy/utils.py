'''
Utility function for model predictions / deployment
'''
import pandas as pd 

def model_play(obs):
    
    if obs.model_home > 0:
        
        if obs.home_closing > obs.model_home:
            
            return("{} {} - Play Value: 3".format(obs.road_team, obs.road_closing))
        
        elif 0 < obs.home_closing < obs.model_home:
            
            return("{} {} - Play Value: 2".format(obs.road_team, obs.road_closing))
        
        else:
            
            return("{} {} - Play Value: 1".format(obs.road_team, obs.road_closing))
        
    else:
        
        if obs.home_closing < obs.model_home:
            
            return("{} {} - Play Value: 3".format(obs.home_team, obs.home_closing))
        
        elif obs.model_home < obs.home_closing < 0:
            
            return("{} {} - Play Value: 2".format(obs.home_team, obs.home_closing))
                   
        else:
            
            return("{} {} - Play Value: 1".format(obs.home_team, obs.home_closing))

            
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






