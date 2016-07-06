import os
from gtts import gTTS
import numpy as np

def to_speech(text):
  tts = gTTS(text=text, lang = 'en')
  return tts

def save(tts, filename):
  tts.save(filename + '.mp3')

data = {'A':158, 'E':9307, 'M':1318, 'R':576, 'T':637, 'N':5707}
labels = ['A','E','M','R','T','N']
values = [158, 9307, 1318, 576, 637, 5707]

bounds = [np.sum(values[:i]) for i in xrange(0, 7)]
print bounds

n = np.sum(values)

print np.sum(values)
print values/np.sum(values, dtype=float)
p_labels = []
t_labels = []


for i in xrange(n):
  a = int(n * np.random.random())
  for j in xrange(6):
    if bounds[j] <= a and a < bounds[j+1]:
      p_labels.append(labels[j])
    if bounds[j] <= i and i < bounds[j+1]:
      t_labels.append(labels[j])

p_labels = np.array(p_labels)
t_labels = np.array(t_labels)

print p_labels[p_labels == t_labels].shape[0] / float(t_labels.shape[0])

