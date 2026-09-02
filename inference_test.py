import torch
import data_func
import models
from utils import Pre_data
from collections import OrderedDict
from tqdm import tqdm
import torch.nn.functional as F
import json
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'data_test' / 'data_new'

model_name = 'Model_51_1'
data_pre_func = 'Data_Func_New'

DEFAULT_CHECKPOINT = PROJECT_ROOT / 'checkpoints' / 'model_recall.chkpt'
batch_size = 1
end_epoch = 7000
device_ids = 0
dropout = 0.1
n_layer_d = 1
n_layer_g = 3

loss_a = 0.7
corr = 1
gamma = 5
alpha = 0.4

parser = argparse.ArgumentParser(description='F-CPI 预训练模型推理')
parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_ROOT)
parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
args = parser.parse_args()
data_root = args.data_dir.resolve()
pretrained_path = args.checkpoint.resolve()

if not pretrained_path.is_file():
    raise FileNotFoundError(f'未找到模型检查点: {pretrained_path}')
if not data_root.is_dir():
    raise FileNotFoundError(f'未找到数据目录: {data_root}')

if args.device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('指定了 CUDA，但当前 PyTorch 无法使用 CUDA')
device = torch.device(
    'cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu'
)

model = getattr(models, model_name)(dropout=dropout, n_layers_d=n_layer_d,
                                    n_gin_layers=n_layer_g)

checkpoint = torch.load(str(pretrained_path), map_location=device)

state_dict = checkpoint['model']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k[:7] == 'module.':
        name = k[7:]
        new_state_dict[name] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict)
model.to(device)

model.eval()
metric = eval('data_func.'+data_pre_func+'_Metric')()
desc = '  - (Testing)   '

data = Pre_data('inf', data_pre_func, batch_size, inf_data=str(data_root))
loaders = data.get_loaders()
inf_loader = loaders['inf']
suc = []
with torch.no_grad():
    for batch in tqdm(inf_loader, mininterval=2, desc=desc, leave=False):
        pro_seq, f_seq, nf_seq, react, gold = batch
        pro_seq = [tensor.to(device) for tensor in pro_seq]
        f_seq = [tensor.to(device) for tensor in f_seq]
        nf_seq = [tensor.to(device) for tensor in nf_seq]
        react = react.to(device)

        pred,_,_ = model(pro_seq, f_seq, nf_seq, react)
        print(pred)
        
        # 应用softmax得到概率分布
        prb = F.softmax(pred, dim=1)
        
        # 存储两个类别的概率
        inactive_prob = prb[0][0].item()  # 活性减弱概率
        active_prob = prb[0][1].item()    # 活性增强概率
        
        suc.append([int(gold[0][0].item()), inactive_prob, active_prob])

print(suc)

# 原有的 positive_pre 文件（保持兼容性）
o = open(data_root / 'positive_pre', 'w')
vocab = [token.strip() for token in open(data_root / 'inf_set')]
o_list = []
for index_prob, inactive_prob, active_prob in suc:
    o_list.append((active_prob, vocab[index_prob]))
o_list = sorted(o_list, key=lambda x:(x[0], x[1]), reverse=True)
print(o_list)

for active_prob, tx in o_list:
    o.write(str(active_prob)+'_'+tx+'\n')
o.close()

print("基础结果已保存到 positive_pre")

# 新增：输出到 result 文件
result_file = data_root / 'result'
with open(result_file, 'w') as result_f:
    results = []
    
    for index_prob, inactive_prob, active_prob in suc:
        tx = vocab[index_prob]
        parts = tx.split('_')
        if len(parts) >= 5:
            f_smiles = parts[0]
            original_smiles = parts[2] 
            protein_info = parts[4] if len(parts) > 4 else "unknown"
            
            # 确定预测结果
            prediction = '活性增强' if active_prob > inactive_prob else '活性减弱'
            
            # 构建结果信息
            result_data = {
                'inactive_probability': round(inactive_prob * 100, 2),  # 活性减弱概率（百分比）
                'active_probability': round(active_prob * 100, 2),      # 活性增强概率（百分比）
                'prediction': prediction,
                'f_smiles': f_smiles,
                'original_smiles': original_smiles,
                'protein_info': protein_info
            }
            results.append(result_data)
    
    # 将结果列表写入文件
    json.dump(results, result_f, indent=2, ensure_ascii=False)

print(f"完整结果已保存到 {result_file}")
print("推理完成")
