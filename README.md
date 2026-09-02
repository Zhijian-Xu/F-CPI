# F-CPI

F-CPI is a local web-based system for fluorine-substitution molecular optimization and activity prediction.

## 1. Environment

Recommended environment:

```powershell
conda create --name fcpi python=3.7 pip -y
conda activate fcpi
conda install setuptools=68.0.0 wheel=0.41.2 -y
$env:SETUPTOOLS_USE_DISTUTILS="stdlib"
python -m pip install -r requirements.txt --index-url https://pypi.org/simple --only-binary=lmdb
```

## 2. Large Files

Large files are provided in the GitHub **Release**:

- `checkpoints.zip` — model checkpoints
- `swissprot.zip` — Swiss-Prot BLAST database
- `ncbi-blast-2.17.0+-win64.exe` — NCBI BLAST+ 2.17.0 for Windows

After downloading:

- Extract `checkpoints.zip` to `F-CPI/checkpoints/`
- Extract `swissprot.zip` to `F-CPI/data_test/blastdb/swissprot/`
- Install NCBI BLAST+ (example path: `D:\blast-2.17.0+`)

## 3. BLAST Configuration

Before running F-CPI, configure the current PowerShell terminal:

```powershell
$env:FCPI_PSIBLAST="D:\blast-2.17.0+\bin\psiblast.exe"
$env:FCPI_BLAST_DB="D:\F-CPI\data_test\blastdb\swissprot\swissprot"
```

These environment variables need to be set again after opening a new terminal.

## 4. Run

```powershell
cd web
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## 5. Functions

F-CPI provides two main functions:

- **Molecular optimization**: input a molecular SMILES and a protein sequence, and return the top fluorine-substituted candidate molecules.
- **Activity prediction**: input the original SMILES, fluorine-substituted SMILES, and protein sequence to predict whether activity is improved.
