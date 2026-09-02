from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# 创建SQLAlchemy实例，用于数据库操作
db = SQLAlchemy()

# 用户表模型
class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
     # 密码哈希值 - 不能为空
    password = db.Column(db.String(255), nullable=False)
    createtime = db.Column(db.DateTime, default=datetime.utcnow)
    isactive = db.Column(db.Boolean, default=True)

    analyses = db.relationship('Analysis', backref='user', lazy=True, cascade='all, delete-orphan')

    # 设置密码方法 - 将明文密码转换为哈希值存储
    def set_password(self, raw_password):
        self.password = generate_password_hash(raw_password)

    # 验证密码方法 - 检查输入的密码是否与存储的哈希值匹配
    def check_password(self, raw_password):
        return check_password_hash(self.password, raw_password)

    # 对象表示方法 - 便于调试和日志输出
    def __repr__(self):
        return f'<User {self.name}>'


# 分析任务表
class Analysis(db.Model):
    __tablename__ = 'analyses'

    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    nfsmiles = db.Column(db.Text, nullable=False)
    fsmiles = db.Column(db.Text, nullable=True)
    proteinsmiles = db.Column(db.Text, nullable=False)
    model = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default='pending')
    createtime = db.Column(db.DateTime, default=datetime.utcnow)

    results = db.relationship('Result', backref='analysis', lazy=True, cascade='all, delete-orphan')


# 分析结果表
class Result(db.Model):
    __tablename__ = 'results'

    id = db.Column(db.Integer, primary_key=True)
    analysisid = db.Column(db.Integer, db.ForeignKey('analyses.id'), nullable=False)
    result = db.Column(db.Text)
    createtime = db.Column(db.DateTime, default=datetime.utcnow)