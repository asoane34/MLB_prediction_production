from flask import Flask, render_template
import pandas as pd
import json
import tensorflow as tf 
from tensorflow.keras.models import model_from_json 
from utils import generate_lines, model_play

today_games = pd.read_csv("../data_collection/all_data/today_games.csv")

model_inputs = pd.read_csv("../data_collection/all_data/model_prepared.csv").values

output_columns = [
    "Date", "Road Team", "Home Team", "Road Starter", "Home Starter", "Home Closing",
    "Road Closing", "Model Plays"
]

def load_model():

    with open("../models/gauss_rank.json", "r+") as f:

        json_model = json.load(f)

    global model
    
    model = model_from_json(json_model)
    
    model.load_weights("../models/gauss_rank_weights.h5")

    print("Model loaded")

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def render_predictions():

    load_model()

    all_picks = generate_lines(model, today_games, model_inputs)

    all_picks["Model Plays"] = all_picks.apply(lambda x: model_play(x), axis = 1)

    all_picks = all_picks.drop(columns = ["model_home"])

    all_picks.columns = output_columns

    return(render_template("picks.html", tables = [all_picks.to_html(classes = "table table-striped table-dark", index = False)]))

if __name__ == "__main__":

    app.run(host="0.0.0.0", port = 5000)
