import os
import shutil
import subprocess
from pathlib import Path

data_dir = Path(__file__).resolve().parents[1]
input_dir = data_dir / 'bin' / 'fasta_data'
tmp_dir = data_dir / 'bin' / 'tmp'
pssm_dir = data_dir / 'bin' / 'pssm'
db_path = os.environ.get('FCPI_BLAST_DB')
blast_bin = os.environ.get('FCPI_PSIBLAST', 'psiblast')

if not db_path:
    raise RuntimeError('请先设置 FCPI_BLAST_DB 环境变量（BLAST 数据库前缀）')

for d in (input_dir, tmp_dir, pssm_dir):
    os.makedirs(d, exist_ok=True)

while os.listdir(input_dir):
    file = os.listdir(input_dir)[0]
    name = file.split('.')[0]
    in_file = input_dir / file
    tmp_pssm = tmp_dir / (name + '.pssm')

    # 关键：把 -out_ascii_pssm 写全，路径加双引号
    cmd = [blast_bin, '-query', str(in_file), '-db', db_path,
           '-out_ascii_pssm', str(tmp_pssm), '-num_threads', '4',
           '-num_iterations', '3', '-evalue', '0.001']
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

    if tmp_pssm.stat().st_size > 0:
        shutil.copy2(tmp_pssm, pssm_dir / tmp_pssm.name)
    if tmp_pssm.exists():
        tmp_pssm.unlink()
    if in_file.exists():
        in_file.unlink()
