'''
Model architecture - retrain model / refit data preparation
'''
import json
import os
import pickle
import pandas as pd 
import numpy as np
from gauss_rank_scaler import GaussRankScaler
from dataclasses import dataclass, field
import tensorflow as tf
from keras import Sequential
from keras.layers import Input, Activation, Dense, Dropout, BatchNormalization
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
'''
Read in original model data + data from current season... 
Frequency of model retraining subject to user
'''
try:

    current_season = pd.read_csv("./data_collection/all_data/model_raw.csv")

except Exception as e:

    print("Could not retrieve current season data: {}".format(e))

    raise

try:
    
    past_seasons = pd.read_csv("./data_collection/all_data/past.csv.gz", compression = "gzip")

except Exception as e:

    print("Could not retrieve past season data: {}".format(e))

    raise

with open("./data_collection/all_data/model_columns.json", "r+") as f:

    model_columns = json.load(f)

'''
Create target variable, current season data
'''
current_season["home_win"] = (current_season.score1 > current_season.score2).astype("uint8")

current_season = current_season[model_columns + ["home_win"]]

train = pd.concat([past_seasons, current_season], axis = 0).reset_index(drop = True)

train_target, train = train["home_win"], train.drop("home_win", axis = 1)

'''
Prepare data for model with GaussRankScaler
'''

GRS = GaussRankScaler()

GRS.fit(train)

if not os.path.exists("./models/"):

    os.makedirs("./models/")

with open("./models/current_scaler.pk", "wb") as f:

    pickle.dump(GRS, f)

train = pd.DataFrame(GRS.transform(train))

train.columns = model_columns

'''
Build Model
'''
@dataclass
class NeuralNetConstructor():
    '''
    Constructor class for Neural Network. 
    '''
    training: np.ndarray
    train_target: np.ndarray
    validation: np.ndarray = None
    validation_target: np.ndarray = None
    external_validation: bool = False
    objective: str = "binary"
    n_classes: int = None
    n_hidden: int = 0
    base_neurons: int = 500
    shape: str = "funnel"
    funnel_param: float = 0.25
    dropout: bool = False
    dropout_pct: float = 0.3
    kernel_initializer: str = "glorot_normal"
    normalize_batches: bool = False
    model_checkpoint: bool = False
    checkpoint_params: dict = field(default_factory = dict)
    early_stopping: bool = False
    early_stopping_params: dict = field(default_factory = dict)
    activation: str = "relu"
    compile_params: dict = field(default_factory = dict)
    fit_params: dict = field(default_factory = dict)
    
    def train_model(self):
        
        self.create_callbacks()
        
        if self.external_validation:
            
                self.fit_params.update({"validation_data" : (self.validation, self.validation_target)})
        
        model = self.create_model_architecture()
        
        history = model.fit(self.training, self.train_target, **self.fit_params)
        
        return(model, history)
    
    def create_model_architecture(self):
        
        input_dim = self.training.shape[1]
        
        model = Sequential()
        
        model.add(Dense(self.base_neurons, kernel_initializer = self.kernel_initializer,
                        activation = self.activation,
                       input_shape = (input_dim,)))
        
        if self.normalize_batches:
            
            model.add(BatchNormalization())
        
        if self.dropout:
            
            model.add(Dropout(self.dropout_pct))
            
        for _ in range(self.n_hidden):
            
            if self.shape == "funnel":
                
                self.base_neurons = int(self.base_neurons * self.funnel_param)
                
                model.add(Dense(self.base_neurons, kernel_initializer = self.kernel_initializer,
                                activation = self.activation))
                        
            elif self.shape == "rectangle":
                
                model.add(Dense(self.base_neurons, kernel_initializer = self.kernel_initializer,
                                activation = self.activation))
                
            else:
                
                raise ValueError("Unsupported network shape provided")
            
            if self.normalize_batches:
                
                model.add(BatchNormalization())
                
            if self.dropout:
                
                model.add(Dropout(self.dropout_pct))
                
        if self.objective == "binary":
            
            model.add(Dense(1, kernel_initializer = self.kernel_initializer,
                            activation = "sigmoid"))
            
        else:
            
            if not self.n_classes:
                
                raise ValueError("If objective is not binary, must pass n_classes")
            
            else:
            
                model.add(Dense(self.n_classes, kernel_initializer = self.kernel_initializer,
                                activation = "softmax"))
        
        model.compile(**self.compile_params)
            
        return(model)
    
    def create_callbacks(self):
        
        if self.early_stopping and self.model_checkpoint:
            
            early_stopping_monitor = EarlyStopping(**self.early_stopping_params)
            
            model_checkpointer = ModelCheckpoint(**self.checkpoint_params)
            
            self.fit_params.update({"callbacks" : [early_stopping_monitor, model_checkpointer]})
            
        elif self.early_stopping:
            
            early_stopping_monitor = EarlyStopping(**self.early_stopping_params)
            
            self.fit_params.update({"callbacks" : [early_stopping_monitor]})
            
        elif self.model_checkpoint:
            
            model_checkpointer = ModelCheckpoint(**self.checkpoint_params)
            
            self.fit_params.update({"callbacks" : [model_checkpointer]})
            
    @staticmethod
    def save_model(model, model_path, weight_path):
        
        json_model = model.to_json()
        
        with open(model_path, "w+") as f:
            
            json.dump(json_model, f)
            
        model.save_weights(weight_path)
        
        print("Model and weights written to file")

if __name__ == "__main__":

    nn_params = {
        
        "training" : train.values,
        
        "train_target" : train_target.values,
        
        "n_hidden" : 2,
        
        "n_classes" : 1,
        
        "objective" : "binary",
        
        "base_neurons" : 250,
        
        "shape" : "rectangle",
        
        "kernel_initializer" : "glorot_normal",
        
        "dropout" : True,
        
        "dropout_pct" : 0.50,
        
        "normalize_batches" : True,
        
        "early_stopping" : True,
        
        "early_stopping_params" : {
            
            "patience" : 100,
            
            "monitor" : "val_loss",
            
            "mode" : "max",
            
            "restore_best_weights" : True
        },
        
        "compile_params" : {
            
            "optimizer" : Adam(),

            "loss" : "binary_crossentropy",

            "metrics" : ["accuracy"]
            
        },
        
        "fit_params" : {
        
            "epochs" : 25,

            "batch_size" : 512
            
        }
        
    }

    nn_model = NeuralNetConstructor(**nn_params)

    print("Training Model...")
    model, history = nn_model.train_model()

    nn_model.save_model(model, "./models/gauss_rank.json", "./models/gauss_rank_weights.h5")
