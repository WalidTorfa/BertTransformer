import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.layers import *
import preprocessing as x
import tensorflow_hub as hub #1
import tensorflow_text as text #2 both are important for the hub
import numpy as np
x_train, x_test, y_train, y_test=x.preprocess()
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")  ## Preprocess padding and tokenizer
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")  ## Model
# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text') #create input layer

preprocessed_text = bert_preprocess(text_input)

outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout", )(inputs = outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)
model = tf.keras.Model(inputs=[text_input], outputs = [l])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=25, batch_size = 64,callbacks = tf.keras.callbacks.TensorBoard(log_dir = "./logs"),validation_data = [x_test,y_test])
y_predicted = model.predict(x_test)
y_predicted = np.round(y_predicted)
print(classification_report(y_predicted, y_test))
model.save("BertModel.h5")
