import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
cleveland = pd.read_csv('input/heart.csv')
print('Shape of DataFrame: {}'.format(cleveland.shape))
print (cleveland.loc[1])

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

sns.heatmap(data.corr(), annot=True, fmt='.1f')
plt.show()
age_unique = sorted(data.age.unique())
age_thalach_values = data.groupby('age')['thalach'].count().values
mean_thalach = []
for i, age in enumerate(age_unique):
    mean_thalach.append(sum(data[data['age'] == age].thalach) / age_thalach_values[i])

X = np.array(data.drop(['target'], 1))
y = np.array(data['target'])
mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])
X_train[0]
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers

Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()

Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1

print(Y_train_binary[:20])


# define a new keras model for binary classification
def create_binary_model():
    # create model
    model = Sequential()
  
    model.add(Dense(16, input_dim=13, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
   
    return model


binary_model = create_binary_model()
binary_model.save('heart1.model')
print(binary_model.summary())

history = binary_model.fit(X_train, Y_train_binary, validation_data=(X_test, Y_test_binary), epochs=100, batch_size=10)
# generate classification report using predictions for categorical model
from sklearn.metrics import classification_report, accuracy_score
# generate classification report using predictions for binary model 
binary_pred = np.round(binary_model.predict(X_test)).astype(int)

print('Results for Binary Model')
print(accuracy_score(Y_test_binary, binary_pred))
print(classification_report(Y_test_binary, binary_pred))

