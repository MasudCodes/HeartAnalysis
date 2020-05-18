import cv2
import tensorflow as tf
import numpy as np;
import sys
import pandas as pd
from numpy import array
import sklearn
import matplotlib
import keras
import joblib
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import * 
import csv

CATEGORIES = ["1", "0"]

model = tf.keras.models.load_model("heart.model")
model.summary()
cleveland = pd.read_csv('input/data.csv')
print('Shape of DataFrame: {}'.format(cleveland.shape))

cleveland.loc[280:]
data = cleveland[~cleveland.isin(['?'])]
data.loc[280:]
data = data.dropna(axis=0)
data.loc[280:]
print(data.shape)
print(data.dtypes)
data = data.apply(pd.to_numeric)
data.dtypes
data.describe()
predictions = model.predict(data)

cat = predictions[0] * 100;

print(predictions)
print(int(predictions[0][0]))

# define and fit the final model

s = model.predict_classes(data)
# Output 1 means yes , output 0 means No
print(s)

border_effects = {
   "flat": tk.FLAT,
     "sunken": tk.SUNKEN,
    "raised": tk.RAISED,
    "groove": tk.GROOVE,
    "ridge": tk.RIDGE,
  }
top = tk.Tk()
# Code to add widgets will go here...

greeting = tk.Label(text="Hello, Tkinter")
label = tk.Label(
    text=(predictions * 100),
    foreground="white",  # Set the text color to white
    background="black"  # Set the background color to black
)

greeting.pack()
button = tk.Button(
    text="Click me!",
    width=25,
    height=5,
    bg="blue",
    fg="yellow",
)

button.pack()
label.pack()
top.mainloop()
