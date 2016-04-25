import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import os
import pickle
import models
import csv

import calculate_features as cf
import models

np.random.seed(200)


def format_time(x, pos=None):
    global duration, nframes, k
    progress = int(x / float(nframes) * duration * k)
    mins, secs = divmod(progress, 60)
    hours, mins = divmod(mins, 60)
    out = "%d:%02d" % (mins, secs)
    if hours > 0:
        out = "%d:" % hours
    return out

def format_db(x, pos=None):
    if pos == 0:
        return ""
    global peak
    if x == 0:
        return "-inf"

    db = 20 * math.log10(abs(x) / float(peak))
    return int(db)

types = {
    1: np.int8,
    2: np.int16,
    4: np.int32
}


def open_wav(path_to_wav, filename):
  wav = wave.open(path_to_wav + filename, mode="r")
  (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
  content = wav.readframes(nframes)

  samples = np.fromstring(content, dtype=types[sampwidth])
  return (nchannels, sampwidth, framerate, nframes, comptype, compname), samples


def get_all_files(path_to_wav):
  files = os.listdir(path_to_wav)
  return files

def get_emotions(path_to_emotions, filename):
  f = open(path_to_emotions + filename, 'r').read()
  # print f
  # print f.split('\n')
  f = np.array(f.split('\n'))#np.append(np.array(['']), np.array(f.split('\n')))
  c = 0
  idx = f == ''
  idx_n = np.arange(len(f))[idx]
  emotion = []
  for i in xrange(len(idx_n) - 2):
    g = f[idx_n[i]+1:idx_n[i+1]]
    head = g[0]
    i0 = head.find(' - ')
    start_time = float(head[head.find('[') + 1:head.find(' - ')])
    end_time = float(head[head.find(' - ') + 3:head.find(']')])
    actor_id = head[head.find(filename[:-4]) + len(filename[:-4]) + 1:head.find(filename[:-4]) + len(filename[:-4]) + 5]
    emo = head[head.find('\t[') - 3:head.find('\t[')]
    vad = head[head.find('\t[') + 1:]

    v = float(vad[1:7])
    a = float(vad[9:15])
    d = float(vad[17:23])
    
    emotion.append({'start':start_time, 
               'end':end_time,
               'id':filename[:-4] + '_' + actor_id,
               'v':v,
               'a':a,
               'd':d,
               'emotion':emo})
  return emotion


def split_wav(wav, emotion):
  (nchannels, sampwidth, framerate, nframes, comptype, compname), samples = wav
  duration = nframes / framerate

  left = samples[0::nchannels]
  right = samples[1::nchannels]

  frames = []
  for ie, e in enumerate(emotions):
    start = e['start']
    end = e['end']

    e['right'] = right[int(start * framerate):int(end * framerate)]
    e['left'] = left[int(start * framerate):int(end * framerate)]

    frames.append({'left':e['left'], 'right': e['right']})
  return frames

available_emotions = ['ang', 'exc', 'fru', 'neu', 'sad']

def get_transcriptions(path_to_transcriptions, filename):
  f = open(path_to_transcriptions + filename, 'r').read()
  f = np.array(f.split('\n'))
  transcription = {}

  for i in xrange(len(f) - 1):
    g = f[i]
    i1 = g.find(': ')
    i0 = g.find(' [')
    ind_id = g[:i0]
    ind_ts = g[i1+2:]
    transcription[ind_id] = ind_ts
  return transcription



data = []
sessions = ['session1', 'session2', 'session3', 'session4', 'session5']
for session in sessions:
  print session
  path_to_wav = 'D:/emotion recognition/' + session + '/dialog/wav/'
  path_to_pics = 'D:/emotion recognition/pics/'
  path_to_emotions = 'D:/emotion recognition/' + session + '/dialog/EmoEvaluation/'
  path_to_transcriptions = 'D:/emotion recognition/' + session + '/dialog/transcriptions/'

  files = get_all_files(path_to_wav)
  files = [f[:-4] for f in files]
  print len(files)
  print files
  for f in files:
    emotions = get_emotions(path_to_emotions, f + '.txt')
    wav = open_wav(path_to_wav, f + '.wav')
    sample = split_wav(wav, emotions)

    transcriptions = get_transcriptions(path_to_transcriptions, f + '.txt')
    for ie, e in enumerate(emotions):
      if e['emotion'] in available_emotions:
        e['left'] = sample[ie]['left']
        e['right'] = sample[ie]['right']
        e['transcription'] = transcriptions[e['id']]
        data.append(e)

def save_sample(x, y, name):
  with open(name, 'w') as csvfile:
    w = csv.writer(csvfile, delimiter=',')
    for i in xrange(x.shape[0]):
      row = x[i, :].tolist()
      row.append(y[i])
      w.writerow(row)

def load(name):
  with open(name, 'r') as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in r:
      x.append(row[:-1])
      y.append(row[-1])
  return np.array(x, dtype=float), np.array(y)

def get_field(data, key):
  return np.array([e[key] for e in data])


def split_sample(n, train_part=0.8):
  ids_idx = np.random.permutation(len(ids))
  step = int(n * (1 - train_part))
  samples = []
  for i in xrange(n / step):
    ids_train = ids[ids_idx[:int(train_part*len(ids)) ]]
    ids_test = ids[ids_idx[int(train_part*len(ids)):]]
    samples.append((ids_train, ids_test))
  return samples


def get_features(data, save=True, path='samples/l_', mode='calculate'):
  if mode == 'calculate':
    for di, d in enumerate(data):
      print di, ' out of ', len(data)
      st_features = cf.calculate_features(d['left'], None).T
      x = []
      y = []
      for f in st_features:
        if f[1] > 1.e-4:
          x.append(f)
          y.append(d['emotion'])
      x = np.array(x, dtype=float)
      y = np.array(y)
      if save:
        save_sample(x, y, path + d['id'] + '.csv')
    return x, y


path_to_probas = 'D:/emotion recognition/probas/'
def plot_probas_emotions(probas, true_emo, filename):
  colors = ['r', 'g', 'b', 'yellow', 'black', 'magenta']
  plt.figure(figsize=(16, 12))
  plt.title('True emotion: ' + true_emo, fontsize=50)
  for j in xrange(probas.shape[1]):
    plt.plot(0.2*np.arange(probas.shape[0]), probas[:, j], color=colors[j], linewidth=3)

  plt.legend(available_emotions, fontsize=20)
  plt.ylim(ymax=1, ymin=0)
  plt.savefig(path_to_probas + filename + '.png')
  plt.close()

from keras.utils import np_utils
def to_categorical(y):
  y_cat = np.zeros((len(y), len(available_emotions)), dtype=int)
  for i in xrange(len(y)):
    y_cat[i, :] = np.array(np.array(available_emotions) == y[i], dtype=int)

  return y_cat


def check_all_finite(X):
  X = np.asanyarray(X)
  if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
          and not np.isfinite(X).all()):
    return True
  else:
    return False

data = np.array(data)
print data
print len(data)

x, y = get_features(data)

ids = get_field(data, 'id')
emotions = get_field(data, 'emotion')

for i in xrange(len(available_emotions)):
  print available_emotions[i], emotions[emotions == available_emotions[i]].shape[0]


parts = 5


permutation = np.random.permutation(len(data))
permuted_ids = ids[permutation]

step = len(data) / parts

preds = []
trues = []

for part in xrange(parts):
  if part >= 0:
    print 'Validation part: ', part
    i0 = step * part
    i1 = step * (part + 1)

    level1_ids = np.append(permuted_ids[:i0], permuted_ids[i1:])
    validation_ids = permuted_ids[i0:i1]

    validation_x = {}
    validation_y = []
    
    for i in validation_ids:
      x, y = load('samples/l_' + i + '.csv')
      validation_x[i] = x
      validation_y.append(y[0])
    validation_y = np.array(validation_y)

    index = 0
    level2_parts = 3
    level2_x = np.zeros((len(data) - len(validation_ids), level2_parts * len(available_emotions)), dtype=np.float32)
    level2_y = []

    for j in xrange(parts - 1):
      print 'Level 1 part: ', j
      j0 = step * j
      j1 = step * (j + 1)

      train_ids = np.append(level1_ids[:j0], level1_ids[j1:])
      test_ids = level1_ids[j0:j1]

      train_x = []
      train_y = []
      test_x = {}
      test_y = {}

      for i in level1_ids:
        x, y = load('samples/l_' + i + '.csv')
        if i in train_ids:
          for s in xrange(x.shape[0]):
            train_x.append(x[s, :])
            train_y.append(y[s])
        if i in test_ids:
          test_x[i] = x
          test_y[i] = y[0]

      train_x = np.array(train_x, dtype=float)
      train_y = np.array(train_y)

      train_y_cat = to_categorical(train_y)

      load_models = False
      if load_models:
        classifiers = pickle.load(open('model' + str(part) + str(j) + '.pkl', 'r'))
      else:
        #classifiers = models.train_rfc(train_x, train_y, None)
        classifiers = models.train_mpc(train_x, train_y_cat, None)
        #pickle.dump(classifiers, open('model' + str(part) + str(j) + '.pkl', 'w'))

      for i in test_ids:
        probs = classifiers.predict(test_x[i])
        for m in xrange(level2_parts):
          m0 = probs.shape[0] * m / level2_parts
          m1 = probs.shape[0] * (m + 1) / level2_parts

          f = np.mean(probs[m0:m1, :], axis=0)
          for fi, ff in enumerate(f):
            if np.isnan(ff):
              f[fi] = 1./float(len(f))
          level2_x[index, m * len(f):(m + 1) * len(f)] = f
        level2_y.append(test_y[i])

        print test_x[i]
        print probs
        print level2_x[index, :]
        print level2_y[index]
        print
        print
        index += 1
    
    #level2_x = level2_x[:len(level2_y), :]
    level2_y = np.array(level2_y)
    print level2_x.shape, level2_y.shape
    print np.sum(level2_x)
    print check_all_finite(level2_x), check_all_finite(level2_y)
    print np.unique(level2_y)

    # for i in xrange(level2_x.shape[0]):
    #   print i, level2_x[i, :], level2_y[i]
    level2_x = np.array(level2_x, dtype=float)
    if load_models:
      level2_classifiers = pickle.load(open('level2_model' + str(part) + '.pkl', 'r'))
    else:
      level2_classifiers = models.train_rfc(level2_x, level2_y, None)
      #pickle.dump(level2_classifiers, open('level2_model' + str(part) + '.pkl', 'w'))

    level1_train_x = []
    level1_train_y = []

    for i in level1_ids:
      x, y = load('samples/l_' + i + '.csv')
      for s in xrange(x.shape[0]):
        level1_train_x.append(x[s, :])
        level1_train_y.append(y[s])
    level1_train_x = np.array(level1_train_x, dtype=float)
    level1_train_y = np.array(level1_train_y)


    level1_train_y_cat = to_categorical(level1_train_y)
    if load_models:
      level1_classifiers = pickle.load(open('level1_model' + str(part) + '.pkl', 'r'))
    else:
      level1_classifiers = models.train_mpc(level1_train_x, level1_train_y_cat, None)
      #pickle.dump(level1_classifiers, open('level1_model' + str(part) + '.pkl', 'w'))



    level2_validation_x = np.zeros((len(validation_ids), level2_parts * len(available_emotions)), dtype=float)
    for ii, i in enumerate(validation_ids):
      probs = level1_classifiers.predict(validation_x[i])
      for m in xrange(level2_parts):
        m0 = probs.shape[0] * m / level2_parts
        m1 = probs.shape[0] * (m + 1) / level2_parts

        f = np.mean(probs[m0:m1, :], axis=0)
        for fi, ff in enumerate(f):
          if np.isnan(ff):
            f[fi] = 1./float(len(f))
        level2_validation_x[ii, m * len(f):(m + 1) * len(f)] = f

    

    prediction = level2_classifiers.predict(level2_validation_x)
    # print pred_probas
    # prediction = []
    # for i in xrange(pred_probas.shape[0]):
    #   prediction.append(available_emotions[np.argmax(pred_probas[i, :])])
    # prediction = np.array(prediction)

    print prediction[prediction == validation_y].shape[0] / float(prediction.shape[0])

    for i in xrange(len(prediction)):
      preds.append(prediction[i])
      trues.append(validation_y[i])

class_to_class_precs = np.zeros((len(available_emotions), len(available_emotions)), dtype=float)

preds = np.array(preds)
trues = np.array(trues)

for cpi, cp in enumerate(available_emotions):
  for cti, ct in enumerate(available_emotions):
    #print cp, ct, emo_pred[(emo_pred == cp) * (emo_test == ct)].shape[0], emo_test[emo_test == ct].shape[0]
    if trues[trues == ct].shape[0] > 0:
      class_to_class_precs[cti, cpi] = preds[(preds == cp) * (trues == ct)].shape[0] / float(trues[trues == ct].shape[0])
    else:
      class_to_class_precs[cti, cpi] = 0.


fig, ax = plt.subplots()
heatmap = ax.pcolor(class_to_class_precs, cmap=plt.cm.Blues)

ax.set_xticklabels(available_emotions, minor=False)
ax.set_yticklabels(available_emotions, minor=False)

ax.set_xticks(np.arange(len(available_emotions)) + 0.5, minor=False)
ax.set_yticks(np.arange(len(available_emotions)) + 0.5, minor=False)

for i in xrange(len(available_emotions)):
  for j in xrange(len(available_emotions)):
    plt.text(i+0.2, j+0.4, str(class_to_class_precs[i, j])[:5])

plt.savefig('level2_precs.png')
plt.close()

# samples = split_sample(len(ids))
# precs = np.zeros(len(samples))

# class_to_class_precs = np.zeros((len(samples), len(available_emotions), len(available_emotions)), dtype=float)

# load_models = True

# level2_parts = 3
# level2_sample = np.zeros((len(data), level2_parts * len(available_emotions)), dtype=float)
# level2_emotions = []


# index = 0
# for it, (ids_train, ids_test) in enumerate(samples):

#   train_x = []
#   test_x = []
#   train_y = []
#   test_y = []

#   test_x2 = {}
#   test_y2 = {}

#   for i in ids:
#     x, y = load('samples/l_' + i + '.csv')
#     if i in ids_train:
#       for j in xrange(x.shape[0]):
#         train_x.append(x[j, :])
#         train_y.append(y[j])
#     if i in ids_test:
#       test_x2[i] = x
#       test_y2[i] = y
#       for j in xrange(x.shape[0]):
#         test_x.append(x[j, :])
#         test_y.append(y[j])

#   train_x = np.array(train_x, dtype=float)
#   train_y = np.array(train_y)
#   test_x = np.array(test_x, dtype=float)
#   test_y = np.array(test_y)

#   if load_models:
#     classifiers = pickle.load(open('model' + str(it) + '.pkl', 'r'))
#   else:
#     classifiers = models.train1(train_x, train_y, None)
#     pickle.dump(classifiers, open('model' + str(it) + '.pkl', 'w'))
#   # classifiers = pickle.load(open('model.pkl', 'r'))

#   for ii, i in enumerate(ids_test):
#     print test_x2[i].shape, i
#     probas = classifiers.predict_proba(test_x2[i])

#     plot_probas_emotions(probas, test_y2[i][0], str(i))

  #   k = probas.shape[0]
  #   l = len(available_emotions)
  #   for j in xrange(level2_parts):
  #     i0 = k * j / level2_parts
  #     i1 = k * (j + 1) / level2_parts

  #     f = np.mean(probas[i0:i1, :], axis=0)
  #     level2_sample[index, j * l:(j + 1) * l] = f
  #   level2_emotions.append(test_y2[i][0])
  #   # print level2_sample[index, :], level2_emotions[index]
  #   index += 1

# level2_emotions = np.array(level2_emotions)

# print len(data)
# print level2_sample.shape
# print level2_emotions.shape

# with open('level2_sample.csv', 'w') as csvfile:
#   w = csv.writer(csvfile, delimiter=',')
#   for i in xrange(level2_sample.shape[0]):
#     d = level2_sample[i, :].tolist()
#     d.append(level2_emotions[i])
#     w.writerow(d)

# #===============================================================================
#   emo_test = []
#   emo_pred = []
#   j = 0
#   for ii, i in enumerate(ids_test):
#     try:
#       predictions_frames = classifiers.predict(test_x2[i])
#       print classifiers.predict_proba(test_x2[i]).shape
#       print classifiers.predict_proba(test_x2[i])
#       ans = np.zeros(len(available_emotions))
#       for ie, e in enumerate(available_emotions):
#         ans[ie] = predictions_frames[predictions_frames == e].shape[0]

#       emo = available_emotions[np.argmax(ans)]
#       emo_test.append(test_y2[i][0])
#       emo_pred.append(emo)
#     except:
#       j += 1
#       print ii, j
#   emo_test = np.array(emo_test)
#   emo_pred = np.array(emo_pred)

#   print emo_pred.shape, emo_test.shape

#   print emo_test[emo_test == emo_pred].shape[0] / float(emo_test.shape[0])
#   precs[it] = emo_test[emo_test == emo_pred].shape[0] / float(emo_test.shape[0])

#   for cpi, cp in enumerate(available_emotions):
#     for cti, ct in enumerate(available_emotions):
#       #print cp, ct, emo_pred[(emo_pred == cp) * (emo_test == ct)].shape[0], emo_test[emo_test == ct].shape[0]
#       if emo_test[emo_test == ct].shape[0] > 0:
#         class_to_class_precs[it, cti, cpi] = emo_pred[(emo_pred == cp) * (emo_test == ct)].shape[0] / float(emo_test[emo_test == ct].shape[0])
#       else:
#         class_to_class_precs[it, cti, cpi] = 0.

# print precs
# print np.mean(precs)

# class_to_class_precs = np.mean(class_to_class_precs, axis=0)

# fig, ax = plt.subplots()
# heatmap = ax.pcolor(class_to_class_precs, cmap=plt.cm.Blues)

# ax.set_xticklabels(available_emotions, minor=False)
# ax.set_yticklabels(available_emotions, minor=False)

# ax.set_xticks(np.arange(len(available_emotions)) + 0.5, minor=False)
# ax.set_yticks(np.arange(len(available_emotions)) + 0.5, minor=False)

# for i in xrange(len(available_emotions)):
#   for j in xrange(len(available_emotions)):
#     plt.text(i+0.2, j+0.4, str(class_to_class_precs[i, j])[:5])

# plt.savefig('precs.png')
# plt.close()

# with open('precs.csv', 'w') as csvfile:
#   w = csv.writer(csvfile, delimiter=',')
#   d = ['']
#   for j in xrange(len(available_emotions)):
#     d.append(available_emotions[j])


#   for i in xrange(len(available_emotions)):
#     d = [available_emotions[i]]
#     for j in xrange(len(available_emotions)):
#       d.append(class_to_class_precs[i, j])
#     w.writerow(d)

# #==============================================================================



# ans, ans_m = models.validate1(classifiers, test_x, test_y)
# print ans_m

# with open('ans.csv', 'w') as csvfile:
#   w = csv.writer(csvfile, delimiter='\t')
#   for r in xrange(ans.shape[0]):
#     print available_emotions[r], ans_m[r, :]
#     w.writerow(ans_m[r, :])





# save(x, y, 'data.csv')

# print np.unique(y)














