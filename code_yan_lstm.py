import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import wave
import numpy as np
import math
import os
import pickle
import models
import csv

import calculate_features as cf
import models

np.random.seed(200)

regime = 'aibo5'

def get_params(regime):
  if regime == 'iemocap':
    #available_emotions = ['ang', 'exc', 'fru', 'neu', 'sad']
    available_emotions = ['ang', 'exc', 'neu', 'sad']
    path_to_samples = 'iem_samples/'
    conf_matrix_prefix = 'iemocap'
    framerate = 44100
    return available_emotions, '', '', '', path_to_samples, conf_matrix_prefix, framerate, '', 0
  elif regime == 'aibo4':   
    available_emotions = ['N', 'M', 'E', 'A']
    
    path_to_wav = 'aibo_data/wav/'
    path_to_transcription = 'aibo_data/transliteration/'
    path_to_labels = 'aibo_data/labels/CEICES/'
    path_to_samples = 'yan_samples_lstm4/'
    conf_matrix_prefix = 'aibo4'
    labels_file = 'word_labels_4cl_aibo_word_set.txt'
    label_pos = 2
    framerate = 16000
    return available_emotions, path_to_wav, path_to_transcription, path_to_labels, path_to_samples, conf_matrix_prefix, framerate, labels_file, label_pos
  else:
    available_emotions = ['N', 'R', 'E', 'A', 'P']
    
    path_to_wav = 'aibo_data/wav/'
    path_to_transcription = 'aibo_data/transliteration/'
    path_to_labels = 'aibo_data/labels/IS2009EmotionChallenge/'
    path_to_samples = 'yan_samples_lstm5/'
    conf_matrix_prefix = 'aibo5'
    labels_file = 'chunk_labels_5cl_corpus.txt'
    framerate = 16000
    label_pos = 1
    return available_emotions, path_to_wav, path_to_transcription, path_to_labels, path_to_samples, conf_matrix_prefix, framerate, labels_file, label_pos

available_emotions, path_to_wav, path_to_transcription, path_to_labels, path_to_samples, conf_matrix_prefix, framerate, labels_file, label_pos = get_params(regime)

segmentation = 'by_phrase'

types = {1: np.int8, 2: np.int16, 4: np.int32}

def open_wav(path_to_wav, filename):
  wav = wave.open(path_to_wav + filename, mode="r")
  (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
  content = wav.readframes(nframes)

  samples = np.fromstring(content, dtype=types[sampwidth])
  return (nchannels, sampwidth, framerate, nframes, comptype, compname), samples

def read_lines(f):
  lines = open(f, 'r').read()
  return np.array(lines.split('\n'))


def get_needed_lines(lines, i, subline):
  result = []
  j = i
  condition = True
  while condition:
    if lines[j][:len(subline)] == subline:
      result.append(lines[j])
      j += 1
    else:
      condition = False
  return result, j

def get_label(expert_grades):
  u = np.unique(expert_grades)
  grade = u[0]
  count = expert_grades[expert_grades == grade].shape[0]
  for ui in u:
    current_count = expert_grades[expert_grades == ui].shape[0]
    if current_count > count:
      grade = ui
      count = current_count
  return grade



def read_aibo_data():
  data = []
  files = np.sort([f[:-4] + '.wav' for f in os.listdir(path_to_wav)])
  transliterations = read_lines(path_to_transcription + 'transliteration.txt')
  labels = read_lines(path_to_labels + labels_file)
  index_t = 0
  index_l = 0

  c = 0
  for fi, f in enumerate(files):
    d = {}
    if fi % 1000 == 0:
      print f, fi, ' out of ', len(files)
    w = open_wav(path_to_wav, f)
    signal = w[1]
    length = w[0][3] / float(w[0][2])
    t, index_t = get_needed_lines(transliterations, index_t, f[:-4])
    l, index_l = get_needed_lines(labels, index_l, f[:-4])
    if l != []:
      grades = []

      em = l[0].split(' ')[label_pos][0]
      d['id'] = f[:-4]
      d['emotion'] = em
      d['signal'] = signal
      d['transcription'] = t[0][14:-2]
      d['length'] = length
      if (d['emotion'] in available_emotions) and (d['length'] > 0.8):
        data.append(d)
    else:
      c += 1

  print 'missed: ', c

  return data


sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

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

def split_wav(wav, emotions):
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

def read_iemocap_data():
  data = []
  for session in sessions:
    print session
    path_to_wav = 'iemocap_data/' + session + '/dialog/wav/'
    path_to_emotions = 'iemocap_data/' + session + '/dialog/EmoEvaluation/'
    path_to_transcriptions = 'iemocap_data/' + session + '/dialog/transcriptions/'

    files = os.listdir(path_to_wav)
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
          e['signal'] = sample[ie]['left']
          #e['right'] = sample[ie]['right']
          e['transcription'] = transcriptions[e['id']]
          data.append(e)
  return data

def read_data():
  if regime == 'aibo4' or regime == 'aibo5':
    return read_aibo_data()
  else:
    return read_iemocap_data()

def check_all_finite(X):
  X = np.asanyarray(X)
  if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
          and not np.isfinite(X).all()):
    return True
  else:
    return False

def to_categorical(y):
  y_cat = np.zeros((len(y), len(available_emotions)), dtype=int)
  for i in xrange(len(y)):
    y_cat[i, :] = np.array(np.array(available_emotions) == y[i], dtype=int)

  return y_cat

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

def get_features(data, save=True, path=path_to_samples):
  failed_samples = []
  for di, d in enumerate(data):
    if di%1000 == 0: 
      print di, ' out of ', len(data)
    st_features = cf.calculate_features(d['signal'], framerate, None).T
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

def get_field(data, key):
  return np.array([e[key] for e in data])


def balance_sample(x, y, size):
  labels = np.unique(y)
  xc = {}
  yc = {}
  for l in labels:
    xc[l] = x[y == l]
    yc[l] = y[y == l]


  s = size / len(labels)

  tx = np.zeros((s*len(labels), x.shape[1]), dtype=float)
  ty = np.zeros(s*len(labels), dtype=str)
  for i in xrange(len(labels)):
    j = i*s
    n = xc[labels[i]].shape[0]

    idx = np.random.randint(low=0, high=n, size=s)
    tx[j:j+s, :] = xc[labels[i]][idx, :]
    ty[j:j+s] = yc[labels[i]][idx]

  return tx, ty

def get_sample(idx, path):
  tx = []
  ty = []
  for i in idx:
    x, y = load(path + '/' + i + '.csv')
    if len(x) < 40:
      tx.append(np.array(x, dtype=float))
      #tx.append(np.array(x, dtype=float))
      ty.append(y[0])

  tx = np.array(tx)
  ty = np.array(ty)
  return tx, ty

def normalize(x):
  gminx = np.zeros(x[0].shape[1]) + 1.e5
  gmaxx = np.zeros(x[0].shape[1]) - 1.e5
  for i in xrange(x.shape[0]):
    q = x[i]
    minx = np.min(q, axis=0)
    maxx = np.max(q, axis=0)

    for s in xrange(x[0].shape[1]):
      if gminx[s] > minx[s]:
        gminx[s] = minx[s]
      if gmaxx[s] < maxx[s]:
        gmaxx[s] = maxx[s]

  for i in xrange(x.shape[0]):
    for s in xrange(x[0].shape[1]):
      x[i][:, s] = (x[i][:, s] - gminx[s]) / float(gmaxx[s] - gminx[s])


  return x

def grow_sample(x, y, n=10000):
  xg = []
  yg = []
  eps = 5.*1.e-2
  for i in xrange(n):
    j = np.random.randint(x.shape[0])
    x0 = x[j]
    x0 += eps * np.random.normal(0, 1, size=x0.shape)
    y0 = y[j]
    xg.append(x0)
    yg.append(y0)
  return np.array(xg), np.array(yg)

def reshape_for_dense(x, y):
  j = 0
  xr = []
  yr = []
  for i in xrange(x.shape[0]):
    for k in xrange(x[i].shape[0]):
      xr.append(x[i][k, :])
      yr.append(y[i])
  return np.array(xr), np.array(yr)


def pad_sequence(x, ts):
  xp = []
  for i in xrange(x.shape[0]):
    x0 = np.zeros((ts, x[i].shape[1]), dtype=float)
    if ts > x[i].shape[0]:
      x0[ts - x[i].shape[0]:, :] = x[i]
    else:
      maxe = np.sum(x[i][0:ts, 1])
      for j in xrange(x[i].shape[0] - ts):
        if np.sum(x[i][j:j + ts, 1]) > maxe:
          x0 = x[i][j:j + ts, :]
          maxe = np.sum(x[i][j:j + ts, 1])
    xp.append(x0)
  return np.array(xp)



data = np.array(read_data())
print data
print len(data)

ids = get_field(data, 'id')
emotions = get_field(data, 'emotion')
print np.unique(emotions)

for i in xrange(len(available_emotions)):
  print available_emotions[i], emotions[emotions == available_emotions[i]].shape[0]

parts = 5
permutation = np.random.permutation(len(data))
permuted_ids = ids[permutation]

step = len(data) / parts

preds = []
trues = []

get_features(data)

for part in xrange(parts):
  i0 = step * part
  i1 = step * (part + 1)

  train_idx = np.append(permuted_ids[:i0], permuted_ids[i1:])
  test_idx = permuted_ids[i0:i1]

  train_x, train_y = get_sample(train_idx, path_to_samples)



  # energies = []
  # for i in xrange(train_x.shape[0]):
  #   energies.append(np.mean(train_x[i][:, 1]))

  # quantile = 0.5
  # largest_energy_idx = np.argsort(energies)[:int(quantile*len(energies))]

  # train_x = train_x[largest_energy_idx]
  # train_y = train_y[largest_energy_idx]

  test_x, test_y = get_sample(test_idx, path_to_samples)

  # train_x = normalize(train_x)
  # test_x = normalize(test_x)

  #train_x, train_y = grow_sample(train_x, train_y, 10000)
  #train_x, train_y = reshape_for_dense(train_x, train_y)
  print train_x.shape
  print train_y.shape

  # lengths = {}
  # for i in xrange(train_x.shape[0]):
  #   if train_x[i].shape[0] in lengths.keys():
  #     lengths[train_x[i].shape[0]] += 1
  #   else:
  #     lengths[train_x[i].shape[0]] = 1

  # for k, v in lengths.items():
  #   print k, v

  ts = 32

  train_x = pad_sequence(train_x, ts)
  test_x = pad_sequence(test_x, ts)

  train_y_cat = to_categorical(train_y)
  test_y_cat = to_categorical(test_y)
  model = models.train_lstm(train_x, train_y_cat, test_x, test_y_cat)

  scores = model.predict(test_x)
  prediction = np.array([available_emotions[np.argmax(t)] for t in scores])
  print prediction[prediction == test_y].shape[0] / float(prediction.shape[0])
  #test_x, test_y = get_sample(train_idx, 'yan_samples_lstm4')

  for i in xrange(len(prediction)):
    preds.append(prediction[i])
    trues.append(test_y[i])

class_to_class_precs = np.zeros((len(available_emotions), len(available_emotions)), dtype=float)

preds = np.array(preds)
trues = np.array(trues)

print 'Total accuracy: ', preds[preds == trues].shape[0] / float(preds.shape[0])

for cpi, cp in enumerate(available_emotions):
  for cti, ct in enumerate(available_emotions):
    #print cp, ct, emo_pred[(emo_pred == cp) * (emo_test == ct)].shape[0], emo_test[emo_test == ct].shape[0]
    if trues[trues == ct].shape[0] > 0:
      class_to_class_precs[cti, cpi] = preds[(preds == cp) * (trues == ct)].shape[0] / float(trues[trues == ct].shape[0])
    else:
      class_to_class_precs[cti, cpi] = 0.


fig, ax = plt.subplots()
heatmap = ax.pcolor(class_to_class_precs.T, cmap=plt.cm.Blues)

ax.set_xticklabels(available_emotions, minor=False)
ax.set_yticklabels(available_emotions, minor=False)

ax.set_xticks(np.arange(len(available_emotions)) + 0.5, minor=False)
ax.set_yticks(np.arange(len(available_emotions)) + 0.5, minor=False)

for i in xrange(len(available_emotions)):
  for j in xrange(len(available_emotions)):
    plt.text(i+0.2, j+0.4, str(class_to_class_precs[i, j])[:5])

plt.savefig(conf_matrix_prefix + 'lstm.png')
plt.close()

