import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import os
import pickle

from sklearn.ensemble import RandomForestClassifier as RFC

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


def train(x, y, options):
  classes = np.unique(y)
  print classes


  classifiers = []
  for c in classes:
    out = np.array([int(i) for i in y==c])
    print c, len(out[out == 1]), len(out[out == 0]), len(out[out == 1]) + len(out[out == 0])



    classifier = RFC(n_estimators=10)
    classifier.fit(x, out)
    classifiers.append(classifier)
  return classifiers

def train_rfc(x, y, options):
  classifier = RFC(n_estimators=100)

  return classifier.fit(x, y)


def train_mpc(x, y, options):
  
  batch_size = 128
  nb_classes = 10
  nb_epoch = 20

  model = Sequential()
  model.add(Dense(512, input_shape=(x.shape[1],)))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(y.shape[1]))
  model.add(Activation('softmax'))

  model.summary()

  model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop())

  train_part = 0.8

  # history = model.fit(x[:int(train_part * x.shape[0]), :], y[:int(train_part * x.shape[0])],
  #                   batch_size=batch_size, nb_epoch=nb_epoch,
  #                   verbose=1, validation_data=(x[int(train_part * x.shape[0]):, :], y[int(train_part * x.shape[0]):]))

  history = model.fit(x, y,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=2, validation_data=(x, y), show_accuracy=True)


  return model

def validate1(classifier, test_x, test_y):
  predictions = classifier.predict(test_x)

  total_acc = predictions[predictions == test_y].shape[0] / float(predictions.shape[0])
  print 'Total acc ', total_acc

  classes = np.unique(test_y)
  print test_y
  print predictions
  ans = []
  for ci, c in enumerate(classes):
    print ci
    
    idx = np.array([ii for ii, i in enumerate(test_y==c)])
    out = test_y[idx]
    pred = predictions[idx]

    tt = pred[(pred == c) * (out == c)].shape[0] / float(out[out == c].shape[0])
    tf = pred[(pred == c) * (out != c)].shape[0] / float(out[out != c].shape[0])
    ft = pred[(pred != c) * (out == c)].shape[0] / float(out[out == c].shape[0])
    ff = pred[(pred != c) * (out != c)].shape[0] / float(out[out != c].shape[0])
    print tt, tf, ft, ff, '\t', out[out == c].shape[0] / float(out.shape[0]), out[out != c].shape[0] / float(out.shape[0])
    tt_tf_ft_ff = [tt, tf, ft, ff]
    ans.append(tt_tf_ft_ff)

  ans_matrix = np.zeros((len(classes), len(classes)), dtype=float)
  for ci, c in enumerate(classes):
    for ci2, c2 in enumerate(classes):
      ans_matrix[ci2, ci] = pred[(pred == c) * (test_y == c2)].shape[0] / float(test_y[test_y == c2].shape[0])


  return np.array(ans, dtype=float), ans_matrix

def validate(classifiers, test_x, test_y):
  print len(classifiers)
  classes = np.unique(test_y)
  ans = []
  for ci, c in enumerate(classes):
    print ci
    out = np.array([int(i) for i in test_y==c])
    predictions = classifiers[ci].predict(test_x)
    tt = predictions[(predictions == 1) * (out == 1)].shape[0] / float(out[out == 1].shape[0])
    tf = predictions[(predictions == 1) * (out == 0)].shape[0] / float(out[out == 0].shape[0])
    ft = predictions[(predictions == 0) * (out == 1)].shape[0] / float(out[out == 1].shape[0])
    ff = predictions[(predictions == 0) * (out == 0)].shape[0] / float(out[out == 0].shape[0])
    print tt, tf, ft, ff, '\t', out[out == 1].shape[0] / float(out.shape[0]), out[out == 0].shape[0] / float(out.shape[0])
    tt_tf_ft_ff = [tt, tf, ft, ff]
    ans.append(tt_tf_ft_ff)

  return np.array(ans, dtype=float)













