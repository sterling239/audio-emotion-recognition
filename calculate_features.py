import wave
import numpy as np
import math
import os
import pickle
import cf


def weighting(x):
  if weighting_type == 'cos':
    return x * np.sin(np.pi * np.arange(len(x)) / (len(x) - 1))
  elif weighting_type == 'hamming':
    return x * (0.54 - 0.46 * np. cos(2. * np.pi * np.arange(len(x)) / (len(x) - 1)))
  else:
    return x

def calculate_features(frames, freq, options):
  n = len(frames)
  window_sec = 0.2
  window_n = int(freq * window_sec)
  use_derivatives = False

  st_f = cf.stFeatureExtraction(frames, freq, window_n, window_n / 2)
  if st_f.shape[1] > 2:
    i0 = 1
    i1 = st_f.shape[1] - 1
    if i1 - i0 < 1:
      i1 = i0 + 1
    if use_derivatives:
      deriv_st_f = np.zeros((st_f.shape[0]*3, i1 - i0), dtype=float)
    else:
      deriv_st_f = np.zeros((st_f.shape[0], i1 - i0), dtype=float)
    for i in xrange(i0, i1):
      i_left = i - 1
      i_right = i + 1
      deriv_st_f[:st_f.shape[0], i - i0] = st_f[:, i]
      if use_derivatives:
        if st_f.shape[1] >= 2:
          deriv_st_f[st_f.shape[0]:st_f.shape[0]*2, i - i0] = (st_f[:, i_right] - st_f[:, i_left]) / 2.
          deriv_st_f[st_f.shape[0]*2:st_f.shape[0]*3, i - i0] = st_f[:, i] - 0.5*(st_f[:, i_left] + st_f[:, i_right])
    return deriv_st_f
  elif st_f.shape[1] == 2:
    deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
    deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
    if use_derivatives:
      deriv_st_f[st_f.shape[0]:st_f.shape[0]*2, 0] = st_f[:, 1] - st_f[:, 0]
      deriv_st_f[st_f.shape[0]*2:st_f.shape[0]*3, 0] = np.zeros(st_f.shape[0])
    return deriv_st_f
  else:
    deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
    deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
    if use_derivatives:
      deriv_st_f[st_f.shape[0]:st_f.shape[0]*2, 0] = np.zeros(st_f.shape[0])
      deriv_st_f[st_f.shape[0]*2:st_f.shape[0]*3, 0] = np.zeros(st_f.shape[0])
    return deriv_st_f

  #mt_f, st_f2 = cf.mtFeatureExtraction(frames, 16000, 2*window_n, 2*window_n, window_n, window_n)
  #print st_f.shape, n / window_n 
  # print st_f2.shape, n / window_n 
  # print mt_f.shape, n / window_n 
  # print mt_f
  # for i in xrange(k0, k1):
  #   if overlapping_type in ['l', 'lr']:
  #     i0 = (i - 1) * window_n
  #   else:
  #     i0 = i * window_n
  #   if overlapping_type in ['r', 'lr']:
  #     i1 = (i + 2) * window_n
  #   else:
  #     i1 = (i + 1) * window_n
    
  #   x = frames[i0:i1]
  #   print len(x), i0, i1, len(frames), n / window_n

  #   # x = weighting(x)





























