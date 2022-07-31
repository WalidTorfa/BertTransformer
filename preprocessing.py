import pandas as pd
from sklearn.model_selection import train_test_split as tts
import tensorflow as tf
import tensorflow_hub as hub#1
import tensorflow_text as text#2 both are important for the hub
import numpy as np 
def preprocess():
    data = pd.read_json(("Sarcasm_Headlines_Dataset.json"), lines= True)
    data = data.drop(["article_link"], axis = 1)
    data.drop(data[data.is_sarcastic == 0].index[-2000:], inplace=True)
    x_train, x_test, y_train, y_test = tts(data["headline"], data["is_sarcastic"], test_size = 0.2)


    return(x_train, x_test, y_train, y_test)
def predict(Userin):
    Model= tf.keras.models.load_model(
       ('BertModel.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer})
    data=Userin
    isit = Model.predict([data])
    isit= np.round(isit)
    return(isit)
