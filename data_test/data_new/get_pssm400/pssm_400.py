import os
import numpy as np
from pathlib import Path

# This file converts pssm to pssm-400 based on ex_SSA.py
data_dir = Path(__file__).resolve().parents[1]
path = data_dir / 'bin' / 'pssm'
anji = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
dict = {}
for file in os.listdir(path):
    vocab = [token.strip() for token in open(path / file)]
    vocab = vocab[3:]
    ls = []
    matrix = []
    for line in vocab:
        if line=='':
            break
        line = line.split()
        ls.append(line[1])
        vec = list(map(float, line[2:22]))
        matrix.append(vec)
    matrix = np.array(matrix)
    mi = np.min(matrix)
    ma = np.max(matrix)
    matrix = (matrix-mi)/(ma-mi)
    pssm_400 = np.array([])
    for an in anji:
        ind = [i for i,x in enumerate(ls) if x==an]
        pssm_400 = np.concatenate((pssm_400,np.sum(matrix[ind],axis=0)))
    dict[file.split('.')[0]]=pssm_400
print(len(dict))
np.savez(data_dir / 'pssm_400.npz', **dict)
