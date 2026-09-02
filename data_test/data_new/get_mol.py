
import numpy as np
from pathlib import Path
from rdkit.Chem import AllChem
from rdkit import Chem



DATA_DIR = Path(__file__).resolve().parent
data_train = [token.strip().split('_') for token in open(DATA_DIR / 'inf_set')]
out_train = open(DATA_DIR / 'mol_inf', 'w')

for line in data_train:
    f = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(line[0]), 4, 512)).tolist()
    nf = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(line[2]), 4, 512)).tolist()
    out_train.write(' '.join(list(map(str,f)))+'_'+' '.join(list(map(str,nf)))+'\n')






