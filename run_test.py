import torch, numpy as np
from scipy.io import loadmat
from combinato_algorithm import CombinatoAlgorithm

mat    = loadmat('simulation_5.mat')
fdata  = mat['data'].ravel().astype('float64') * 100
sr     = float(mat['sr'].ravel()[0]) if 'sr' in mat else 24000.0
atimes = np.linspace(0, fdata.shape[0] / (sr / 1000), fdata.shape[0])

signal = torch.tensor(fdata,  dtype=torch.float64)
times  = torch.tensor(atimes, dtype=torch.float64)

model = CombinatoAlgorithm(spc_path=r'C:\Users\hp\Downloads\combinato\spc')
with torch.no_grad():
    result = model(signal, times)

print('pos clusters:', len(set(result['pos_sort_idx']) - {0}))
print('neg clusters:', len(set(result['neg_sort_idx']) - {0}))
print('pos artifacts:', result['pos_artifact_ids'])
print('neg artifacts:', result['neg_artifact_ids'])
print('timings:', result['timings'])