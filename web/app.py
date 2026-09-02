import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data_test' / 'data_new'
TAPE_MODEL_DIR = PROJECT_ROOT / 'checkpoints' / 'tape_models'
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from database import db, User, Analysis, Result   
from datetime import datetime
import subprocess
import sys
import torch
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import copy
from rdkit import Chem
from rdkit.Chem import AllChem

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
db_path = os.path.join(app.instance_path, 'FluoroXplorer.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

class DataProcessor:
    """数据预处理类"""
    
    def __init__(self):
        self.base_dir = str(DATA_DIR)
    
    def create_inf_set_tab2(self, original_smiles, f_substituted_smiles, protein_sequence, protein_name="protein"):
        """为Tab2创建inf_set文件"""
        inf_set_path = os.path.join(self.base_dir, 'inf_set')
        
        # Tab2格式: F取代分子SMILES_分子名称_原分子SMILES_分子名称_蛋白质名称_蛋白质氨基酸序列
        line = f"{f_substituted_smiles}_molf_{original_smiles}_molnf_{protein_name}_{protein_sequence}"
        
        with open(inf_set_path, 'w') as f:
            f.write(line + '\n')
        
        print(f"✅ Tab2 inf_set文件已创建: {inf_set_path}")
        return True
    
    def create_fasta_file(self, protein_sequence, protein_name="9AYL"):
        """创建fasta文件"""
        fasta_dir = os.path.join(self.base_dir, 'bin', 'fasta_data')
        os.makedirs(fasta_dir, exist_ok=True)
        
        fasta_file = os.path.join(fasta_dir, f'{protein_name}.fasta')
        with open(fasta_file, 'w') as f:
            f.write(f">{protein_name}\n{protein_sequence}\n")
        
        print(f"✅ Fasta文件已创建: {fasta_file}")
        return fasta_file
    
    def process_molecules(self):
        """处理分子数据，生成分子指纹"""
        try:
            get_mol_script = os.path.join(self.base_dir, 'get_mol.py')
            
            print(f"运行分子处理脚本: {get_mol_script}")
            
            result = subprocess.run([sys.executable, get_mol_script],
                                 cwd=self.base_dir, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
            
            if result.returncode == 0:
                print("✅ 分子指纹处理成功")
                # 检查生成的mol_inf文件
                mol_inf_file = os.path.join(self.base_dir, 'mol_inf')
                if os.path.exists(mol_inf_file):
                    with open(mol_inf_file, 'r') as f:
                        lines = f.readlines()
                        print(f"✅ 生成了{len(lines)}行分子指纹数据")
                return True
            else:
                error_msg = result.stderr.decode('utf-8') if result.stderr else '未知错误'
                print(f"❌ 分子指纹处理失败: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ 分子处理异常: {str(e)}")
            return False
    
    def process_protein_pssm(self):
        """处理蛋白质PSSM特征"""
        try:
            # 运行MSA处理
            ex_msa_script = os.path.join(self.base_dir, 'get_pssm400', 'ex_MSA.py')
            print(f"运行MSA处理脚本: {ex_msa_script}")
            
            result_msa = subprocess.run([sys.executable, ex_msa_script],
                                     cwd=self.base_dir, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)
            
            if result_msa.returncode != 0:
                error_msg = result_msa.stderr.decode('utf-8', errors='replace') if result_msa.stderr else '未知错误'
                print(f"❌ MSA处理失败: {error_msg}")
                return False
            
            # 运行PSSM处理
            pssm_script = os.path.join(self.base_dir, 'get_pssm400', 'pssm_400.py')
            print(f"运行PSSM处理脚本: {pssm_script}")
            
            result_pssm = subprocess.run([sys.executable, pssm_script],
                                      cwd=self.base_dir, 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)
            
            if result_pssm.returncode == 0:
                print("✅ PSSM处理成功")
                pssm_file = os.path.join(self.base_dir, 'pssm_400.npz')
                if os.path.exists(pssm_file):
                    print(f"✅ 生成了PSSM特征文件: {pssm_file}")
                return True
            else:
                error_msg = result_pssm.stderr.decode('utf-8', errors='replace') if result_pssm.stderr else '未知错误'
                print(f"❌ PSSM处理失败: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ 蛋白质PSSM处理异常: {str(e)}")
            return False
    
    def process_protein_tape(self, protein_name="9AYL"):
        """使用TAPE生成蛋白质特征"""
        try:
            fasta_file = os.path.join(self.base_dir, 'bin', 'fasta_data', f'{protein_name}.fasta')
            output_file = os.path.join(self.base_dir, 'inf_pro_feature.npz')
            tape_config = TAPE_MODEL_DIR / 'config.json'
            tape_weights = TAPE_MODEL_DIR / 'pytorch_model.bin'

            if not tape_config.is_file() or not tape_weights.is_file():
                missing = [str(path) for path in (tape_config, tape_weights) if not path.is_file()]
                print(f"❌ 缺少本地TAPE模型文件: {', '.join(missing)}")
                return False
             
            command = [
                'tape-embed', 'transformer', 
                fasta_file, 
                output_file, 
                str(TAPE_MODEL_DIR),
                '--batch_size=1'
            ]

            if not torch.cuda.is_available():
                command.append('--no_cuda')
            
            print(f"运行TAPE嵌入命令: {' '.join(command)}")
            
            result = subprocess.run(command, 
                                 cwd=self.base_dir, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 env=os.environ.copy())
            
            if result.returncode == 0:
                print("✅ TAPE蛋白质特征提取成功")
                if os.path.exists(output_file):
                    print(f"✅ 生成了蛋白质特征文件: {output_file}")
                return True
            else:
                error_msg = result.stderr.decode('utf-8') if result.stderr else '未知错误'
                print(f"❌ TAPE蛋白质特征提取失败: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ TAPE蛋白质处理异常: {str(e)}")
            return False
    
    def enumerate_hf_substitutions(self, mol_smiles: str, top_k=5):
        """返回 Top-k 个 H→F 取代 SMILES（兼容旧版 RDKit）"""
        mol = Chem.MolFromSmiles(mol_smiles)
        if not mol:
            return []

        # 把隐式氢变成显式氢，才能找到 H 原子
        mol = Chem.AddHs(mol)
        candidates = []

        for idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 1:  # 氢原子
                new_mol = Chem.RWMol(mol)
                new_mol.ReplaceAtom(idx, Chem.Atom(9))  # 换成 F
                # new_mol.ClearAtomProps(idx)  # ← 删除或注释掉
                Chem.SanitizeMol(new_mol)
                f_smiles = Chem.MolToSmiles(new_mol)
                candidates.append(f_smiles)

        candidates = list(set(candidates))  # 去重
        return candidates[:top_k]

class ModelService:
    """模型服务类"""
    
    def __init__(self):
        print("✅ 模型服务初始化完成")
    
    def _get_interpretation(self, score):
        if score >= 0.7:
            return "氟取代可显著增强结合活性（概率≥70%）"
        elif score >= 0.6:
            return "氟取代有较大可能增强结合活性（概率≥60%）"
        elif score >= 0.5:
            return "模型输出已越过0.5决策阈值，氟取代可能显著增强结合活性"
        elif score >= 0.4:
            return "模型输出接近决策边界，轻微倾向于氟取代可能减弱结合活性（概率≈40-50%）"
        elif score >= 0.3:
            return "模型认为氟取代有较大可能减弱结合活性（概率≥30%）"
        else:
            return "模型高度置信氟取代可显著减弱结合活性（概率≥70%）"
        
    def _get_confidence_level(self, score):
        """根据分数确定置信水平"""
        if score >= 0.8 or score <= 0.2:
            return "高"
        elif score >= 0.7 or score <= 0.3:
            return "中"
        else:
            return "低"
    
    def run_inference(self):
        """运行模型推理"""
        try:
            inference_script = str(PROJECT_ROOT / 'inference_test.py')
            
            print(f"运行推理脚本: {inference_script}")
            print(f"工作目录: {PROJECT_ROOT}")
            
            # 设置环境变量
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'
            
            result = subprocess.run([sys.executable, inference_script],
                                 cwd=str(PROJECT_ROOT),
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True,
                                 env=env)
            
            if result.returncode == 0:
                print("✅ 模型推理成功")
                return self._parse_inference_result()
            else:
                error_msg = result.stderr or result.stdout or '未知错误'
                print(f"❌ 模型推理失败: {error_msg}")
                return {
                    "type": "error",
                    "message": f"模型推理失败: {error_msg}"
                }
                
        except Exception as e:
            print(f"❌ 模型推理异常: {str(e)}")
            return {
                "type": "error",
                "message": f"模型推理异常: {str(e)}"
            }

    def _parse_inference_result(self):
        """解析推理结果"""
        try:
            # 优先使用新的 result 文件
            result_file = str(DATA_DIR / 'result')
            positive_pre_file = str(DATA_DIR / 'positive_pre')
            
            # 检查新的 result 文件是否存在
            if os.path.exists(result_file):
                return self._parse_new_result_format(result_file)
            # 回退到旧的 positive_pre 文件
            elif os.path.exists(positive_pre_file):
                return self._parse_legacy_format(positive_pre_file)
            else:
                return {
                    "type": "error",
                    "message": "未找到预测结果文件"
                }
                
        except Exception as e:
            return {
                "type": "error",
                "message": f"解析预测结果失败: {str(e)}"
            }

    def _parse_new_result_format(self, result_file):
        try:
            import json
            with open(result_file, 'r') as f:
                results = json.load(f)

            if not results:
                return {
                    "type": "error",
                    "message": "预测结果文件为空"
                }

            # 取第一个（最高分）结果
            result_data = results[0]
            active_prob = result_data.get('active_probability', 0) / 100.0
            inactive_prob = result_data.get('inactive_probability', 0) / 100.0

            return {
                'type': 'activity_test',
                'prediction': '活性增强' if active_prob > inactive_prob else '活性减弱',
                'active_probability': round(active_prob * 100, 2),
                'inactive_probability': round(inactive_prob * 100, 2),
                'confidence_score': max(active_prob, inactive_prob),
                'interpretation': self._get_interpretation(max(active_prob, inactive_prob)),
                'success': True
            }

        except Exception as e:
            return {
                "type": "error",
                "message": f"解析新格式结果失败: {str(e)}"
            }
    
    def _parse_legacy_format(self, positive_pre_file):
        """解析旧格式的结果"""
        with open(positive_pre_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return {
                "type": "error", 
                "message": "预测结果文件为空"
            }
        
        first_line = lines[0].strip()
        parts = first_line.split('_', 1)
        
        if len(parts) < 2:
            return {
                "type": "error",
                "message": "预测结果格式错误"
            }
        
        score = float(parts[0])
        
        # 计算两个类别的概率
        active_prob = score  # 活性增强的概率
        inactive_prob = 1 - score  # 活性减弱的概率
        
        # 解析文本获取信息
        text_parts = parts[1].split('_')
        if len(text_parts) >= 5:
            f_smiles = text_parts[0]
            original_smiles = text_parts[2]
            protein_info = text_parts[4]
        else:
            f_smiles = original_smiles = protein_info = "解析失败"
        
        return {
            'type': 'activity_test',
            'prediction': '活性增强' if score > 0.5 else '活性减弱',
            'active_probability': round(active_prob * 100, 2),
            'inactive_probability': round(inactive_prob * 100, 2),
            'confidence_score': score,
            'interpretation': self._get_interpretation(score),
            'f_smiles': f_smiles,
            'original_smiles': original_smiles,
            'protein_info': protein_info,
            'success': True
        }

    def predict_candidates(self, original_smiles, protein_seq, candidates):
        """
        对每一条 (original, F-sub, protein) 跑模型，返回 [{'smiles':.., 'prob':..}, ..]
        """
        results = []
        for f_smiles in candidates:
            # 构造单次输入
            input_data = {
                'nfsmiles': original_smiles,
                'fsmiles': f_smiles,
                'proteinsmiles': protein_seq
            }
            # 直接复用 Tab2 全流程
            res = halogen_analysis_model(input_data)
            if res.get('type') == 'activity_test':
                prob = res['active_probability']
                results.append({'smiles': f_smiles, 'prob': prob})
        # 按概率降序
        results.sort(key=lambda x: x['prob'], reverse=True)
        return results

# 创建全局实例
data_processor = DataProcessor()
model_service = ModelService()

def halogen_analysis_model(input_data: dict):
    """氟取代分析模型（Tab2专用）"""
    try:
        nfsmiles = input_data.get('nfsmiles', '').strip()
        fsmiles = input_data.get('fsmiles', '').strip()
        proteinsmiles = input_data.get('proteinsmiles', '').strip()
        
        # 检查输入数据
        if not all([nfsmiles, fsmiles, proteinsmiles]):
            return {
                "type": "error",
                "message": "请输入完整的数据：原分子SMILES、F取代分子SMILES和蛋白质序列",
                "confidence": 0.0
            }
        
        print("=" * 50)
        print("开始Tab2活性测试流程")
        print(f"原分子SMILES: {nfsmiles}")
        print(f"F取代分子SMILES: {fsmiles}")
        print(f"蛋白质序列长度: {len(proteinsmiles)}")
        print("=" * 50)
        
        # 步骤1: 数据存储
        print("步骤1: 数据存储...")
        if not data_processor.create_inf_set_tab2(nfsmiles, fsmiles, proteinsmiles):
            return {
                "type": "error",
                "message": "创建inf_set文件失败",
                "confidence": 0.0
            }
        
        if not data_processor.create_fasta_file(proteinsmiles):
            return {
                "type": "error",
                "message": "创建fasta文件失败",
                "confidence": 0.0
            }
        
        # 步骤2: 数据预处理
        print("步骤2: 数据预处理...")
        
        # 2.1 分子处理
        print("2.1 分子指纹处理...")
        if not data_processor.process_molecules():
            return {
                "type": "error",
                "message": "分子指纹处理失败",
                "confidence": 0.0
            }
            
        # 2.2 蛋白质TAPE特征提取
        print("2.2 蛋白质TAPE特征提取...")
        if not data_processor.process_protein_tape():
            return {
                "type": "error",
                "message": "蛋白质TAPE特征提取失败",
                "confidence": 0.0
            }
         
        # 2.3 蛋白质PSSM处理
        print("2.3 蛋白质PSSM处理...")
        if not data_processor.process_protein_pssm():
            return {
                "type": "error",
                "message": "蛋白质PSSM处理失败",
                "confidence": 0.0
            }
        
        # 步骤3: 模型推理
        print("步骤3: 模型推理...")
        result = model_service.run_inference()
        
        if result:
            print("✅ 活性测试完成")
            return result
        else:
            return {
                "type": "error",
                "message": "模型推理失败",
                "confidence": 0.0
            }
        
    except Exception as e:
        print(f"❌ 活性测试异常: {str(e)}")
        return {
            "type": "error", 
            "message": f"活性测试异常: {str(e)}",
            "confidence": 0.0
        }

def molecule_optimization_model(input_data: dict):
    """
    Tab1：枚举所有 H→F → 批量预测 → Top-5
    使用与 Tab2 相同的模型接口
    """
    try:
        original_smiles = input_data.get('nfsmiles', '').strip()
        protein_seq = input_data.get('proteinsmiles', '').strip()

        if not original_smiles or not protein_seq:
            return {"type": "error", "message": "请输入原分子与蛋白质序列"}

        # 1. 枚举所有 H→F
        candidates = data_processor.enumerate_hf_substitutions(original_smiles, top_k=10)
        if not candidates:
            return {"type": "error", "message": "未找到可取代的H位点"}

        # 2. 批量预测
        ranked = []
        for f_smiles in candidates:
            # 构造单次输入（复用 Tab2 流程）
            input_single = {
                'nfsmiles': original_smiles,
                'fsmiles': f_smiles,
                'proteinsmiles': protein_seq
            }
            res = halogen_analysis_model(input_single)
            if res.get('type') == 'activity_test':
                ranked.append({
                    'smiles': f_smiles,
                    'prob': res['active_probability']
                })

        # 3. 排序并取 Top-5
        ranked.sort(key=lambda x: x['prob'], reverse=True)
        top5 = ranked[:5]

        return {
            "type": "molecule_optimization",
            "candidates": top5
        }

    except Exception as e:
        return {"type": "error", "message": f"枚举预测异常: {str(e)}"}

def model_b(input_data: dict):
    return {"message": f"Model B processed: {input_data}"}

models = {
    "Halogen Analysis": halogen_analysis_model,
    "Model B": model_b,
    "Molecule Optimization": molecule_optimization_model  
}

# ---------- 路由 ----------
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(name=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.name
            flash('登录成功！', 'success')
            return redirect(url_for('index'))
        else:
            flash('用户名或密码错误', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm = request.form['confirm_password']
        email = request.form.get('email', '').strip()
        if password != confirm:
            flash('密码确认不匹配', 'error')
            return render_template('register.html')
        if User.query.filter_by(name=username).first():
            flash('用户名已存在', 'error')
            return render_template('register.html')
        if not email:
            flash('请输入邮箱地址', 'error')
            return render_template('register.html')
        user = User(name=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('注册成功！请登录', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('已成功退出登录', 'success')
    return redirect(url_for('login'))

@app.route('/process', methods=['POST'])
def process():
    if 'user_id' not in session:
        return jsonify({'error': '请先登录'}), 401
    
    data = request.json
    model_name = data.get('model')
    input_data = {
        'nfsmiles': data.get('nfsmiles'),
        'fsmiles': data.get('fsmiles'),
        'proteinsmiles': data.get('proteinsmiles')
    }
    
    # 根据模型类型决定必填字段
    if model_name == 'Molecule Optimization':
        required_fields = ['nfsmiles', 'proteinsmiles']
    else:
        required_fields = ['nfsmiles', 'fsmiles', 'proteinsmiles']

    for field in required_fields:
        if not input_data.get(field):
            return jsonify({'error': f'请输入完整的 {field} 数据'}), 400
    
    if model_name not in models:
        return jsonify({'error': '无效的模型选择'}), 400

    try:
        # 创建分析记录
        analysis = Analysis(
            uid=session['user_id'],
            nfsmiles=input_data['nfsmiles'],
            fsmiles=input_data['fsmiles'],
            proteinsmiles=input_data['proteinsmiles'],
            model=model_name,
            status='processing'
        )
        db.session.add(analysis)
        db.session.flush()

        # 调用模型
        result = models[model_name](input_data)

        # 保存结果
        result_entry = Result(analysisid=analysis.id, result=str(result))
        db.session.add(result_entry)

        # 更新状态
        analysis.status = 'success'
        db.session.commit()
        return jsonify({'output_data': result})
    except Exception as e:
        db.session.rollback()
        analysis.status = 'error'
        db.session.commit()
        return jsonify({'error': f'处理错误: {str(e)}'}), 500

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    records = Analysis.query.filter_by(uid=session['user_id'])\
                             .order_by(Analysis.createtime.desc()).all()
    return render_template('history.html', records=records)

# ---------- 初始化 ----------
def init_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(name='test').first():
            test = User(name='test', email='test@example.com')
            test.set_password('test123')
            db.session.add(test)
            db.session.commit()
            print("测试用户已创建：test / test123")

if __name__ == '__main__':
    init_db()
    app.run(
        host=os.environ.get('FCPI_HOST', '127.0.0.1'),
        port=int(os.environ.get('FCPI_PORT', '5000')),
        debug=os.environ.get('FCPI_DEBUG', '0') == '1'
    )
    # app.run(
    #     host='0.0.0.0',  # host='0.0.0.0' 允许所有IP访问
    #     port=5000,      # port指定端口
    #     debug=False,  # 关闭调试模式
    #     threaded=True  # 启用多线程
    # )
