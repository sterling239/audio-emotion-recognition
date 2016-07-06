import wave
import numpy as np
import math
import os
import pickle
from keras.preprocessing import sequence
from sklearn.ensemble import RandomForestRegressor as RFR

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, TimeDistributedDense
from keras.layers import LSTM
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2
from keras.optimizers import SGD

import h5py

def train_rfr(x, y, tx, ty):
  rfr = RFR(n_estimators=50)
  model = rfr.fit(x, y)

  return model



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

def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks


def make_sample_lstm(x, n, y=None, use_y=False):
  if use_y:
    xt = np.zeros((x.shape[0], n, x.shape[1] + 1), dtype=float)
    t = np.zeros((x.shape[0], x.shape[1] + 1), dtype=float)
    for i in xrange(x.shape[0]):
      if i == 0:
        t[i, :-1] = x[i, :]
        t[i, -1] = 0
      else:
        t[i, :-1] = x[i, :]
        t[i, -1] = y[i-1]

    for i in xrange(x.shape[0]):
      if i < n:
        i0 = n - i
        xt[i, :i0, :] = np.zeros((i0, x.shape[1] + 1), dtype=float)
        if i > 0:
          xt[i, i0:, :] = t[:i, :]
      else:
        xt[i, :, :] = t[i-n:i, :]
    return xt
  else:
    xt = np.zeros((x.shape[0], n, x.shape[1]), dtype=float)
    for i in xrange(x.shape[0]):
      if i < n:
        i0 = n - i
        xt[i, :i0, :] = np.zeros((i0, x.shape[1]), dtype=float)
        if i > 0:
          xt[i, i0:, :] = x[:i, :]
      else:
        xt[i, :, :] = x[i-n:i, :]
    return xt


class modelLSTM:
  def __init__(self, model, length, use_y):
    self.model = model
    self.n = length
    self.use_y = use_y
  def predict(self, x):
    if self.use_y:
      result = np.zeros((x.shape[0], 1), dtype=float)
      for i in xrange(x.shape[0]):
        t = np.zeros((self.n, x.shape[1] + 1), dtype=float)


    else:
      xt = make_sample_lstm(x, self.n)
      return self.model.predict(xt)
  def save(self, name_json, name_weights):
    json_string = self.model.to_json()
    open(name_json, 'w').write(json_string)
    self.model.save_weights(name_weights)


def train_lstm_avec(x, y, xt, yt):
  length = 25
  use_y = False

  x_series_train = make_sample_lstm(x, length, y, use_y)
  print x_series_train.shape
  x_series_test = make_sample_lstm(xt, length, yt, use_y)
  print x_series_test.shape

  print y[:100, 0]
  model = Sequential()
  model.add(LSTM(256, return_sequences=True, input_shape=(x_series_train.shape[1], x_series_train.shape[2])))
  model.add(Dropout(0.2))
  model.add(Activation('tanh'))
  model.add(LSTM(128))
  model.add(Dropout(0.2))
  model.add(Activation('tanh'))
  # model.add(Dense(128))
  # model.add(Activation('tanh'))
  model.add(Dense(1))
  #model.add(Activation('softmax'))  
  model.summary()
  model.compile(loss='mean_absolute_error', optimizer='rmsprop')
  model.fit(x_series_train, y, batch_size=512, nb_epoch=50,
                       verbose=2, validation_data=(x_series_test, yt))

  return modelLSTM(model, length, use_y)





def train_lstm(x, y, xt, yt):
  batch_size = 180
  nb_classes = 10
  nb_epoch = 25
  ts = x[0].shape[0]
  model = Sequential()
  model.add(LSTM(512, return_sequences=True, input_shape=(ts, x[0].shape[1])))
  model.add(Activation('tanh'))
  # model.add(LSTM(512, return_sequences=True))
  # model.add(Activation('tanh'))
  model.add(LSTM(256, return_sequences=False))
  model.add(Activation('tanh'))
  model.add(Dense(512))
  model.add(Activation('tanh'))
  model.add(Dense(y.shape[1]))
  model.add(Activation('softmax'))

  model.summary()

  model.compile(loss='categorical_crossentropy', optimizer=RMSprop())



  # for epoch in xrange(nb_epoch):
  model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,
                       verbose=2, validation_data=(xt, yt), show_accuracy=True)

  return model

def train_mpc(x, y, tx, ty):
  
  batch_size = 256
  nb_classes = 10
  nb_epoch = 20

  model = Sequential()
  model.add(Dense(512, input_shape=(x.shape[1],)))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1024))
  model.add(Activation('relu'))
  model.add(Dense(1024))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dense(1024))
  model.add(Activation('relu'))
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dense(y.shape[1]))
  #model.add(Activation('softmax'))

  #model.summary()

  model.compile(loss='mean_absolute_error',
              optimizer=RMSprop())

  history = model.fit(x, y,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=0, validation_data=(tx, ty), show_accuracy=True)

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


class Random:
  def __init__(self):
    1
  def fit(self, x, y):
    1
  def predict(self, x):
    return np.random.normal(0, 0.1, size=x.shape[0])


def train_rnd(x, y, tx, ty):
  return Random()


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













