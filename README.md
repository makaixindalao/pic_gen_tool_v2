# 作战环境下军事目标数据集生成与管理平台

本项目旨在根据《需求规格说明书》和《项目架构与开发计划》构建一个功能完善的数据集生成与管理平台。

## 项目结构概览

```
.
├── backend/                # 后端服务器 (FastAPI, Celery)
│   ├── app/                # 主应用包
│   │   ├── api/            # API 接口 (Routers)
│   │   │   └── v1/
│   │   │       ├── traditional/     # 传统生成API
│   │   │       └── ai_generation/   # AI生成API
│   │   ├── core/           # 核心逻辑、配置、Celery实例
│   │   ├── crud/           # 数据库交互函数 (Create, Read, Update, Delete)
│   │   ├── db/             # SQLAlchemy 模型和会话管理
│   │   ├── schemas/        # Pydantic 数据模型
│   │   │   ├── traditional/         # 传统生成数据模型
│   │   │   └── ai_generation/       # AI生成数据模型
│   │   ├── services/       # 业务逻辑服务
│   │   │   ├── traditional/         # 传统蒙版合成服务
│   │   │   ├── ai_generation/       # AI生成服务
│   │   │   └── common/              # 通用服务(标注、文件管理等)
│   │   └── tasks/          # Celery 任务定义
│   │       ├── traditional/         # 传统生成任务
│   │       └── ai_generation/       # AI生成任务
│   ├── tests/              # 后端测试
│   └── requirements.txt    # 后端Python依赖
├── data/                   # 生成的数据和原始素材
│   ├── datasets/           # 存储生成的数据集
│   └── raw_assets/         # 存储甲方提供的原始图片素材
│       ├── targets/        # 军事目标图像(坦克、战机、舰艇)
│       └── backgrounds/    # 背景场景图像
├── docs/                   # 项目文档
│   ├── 需求规格说明书.md
│   ├── 项目架构与开发计划.md
│   ├── 传统生成开发计划.md
│   └── AI生成开发计划.md
├── frontend/               # 前端GUI应用 (PyQt)
│   ├── src/                # 源代码
│   │   ├── api_client/     # 与后端API通信的客户端
│   │   └── widgets/        # 可复用的UI组件
│   ├── assets/             # GUI静态资源 (图标等)
│   ├── ui/                 # Qt Designer 生成的 .ui 文件
│   └── requirements.txt    # 前端Python依赖
├── models/                 # AI模型文件
│   ├── stable_diffusion/   # 基础Stable Diffusion模型
│   └── finetuned/          # 微调后的定制化模型
├── scripts/                # 工具和训练脚本
├── .gitignore              # Git忽略配置
├── README.md               # 项目总览
└── requirements.txt        # 项目级或用于安装脚本的依赖
```

## 后端服务架构

### 传统生成服务
- **蒙版合成器** (`services/traditional/mask_compositor.py`): 实现目标图像与背景的精确叠加
- **图像处理器** (`services/traditional/image_processor.py`): 处理光照、风格调整等后期效果
- **传统生成任务** (`tasks/traditional/mask_generation_task.py`): Celery异步任务处理

### AI生成服务  
- **SD生成器** (`services/ai_generation/sd_generator.py`): Stable Diffusion模型调用和管理
- **提示词构建器** (`services/ai_generation/prompt_builder.py`): 自动构建和优化Prompt
- **AI生成任务** (`tasks/ai_generation/sd_generation_task.py`): Celery异步任务处理

### 通用服务
- **标注生成器** (`services/common/annotation_generator.py`): 生成COCO格式标注文件
- **文件管理器** (`services/common/file_manager.py`): 数据集文件组织和存储

## 快速开始

### 环境要求
- Python 3.8+
- PyQt6（前端GUI）
- PyTorch + Diffusers（AI生成）
- FastAPI（后端API）
- 8GB+ 内存（推荐16GB）
- NVIDIA GPU（可选，用于AI生成加速）

### 安装步骤

#### 1. 克隆项目
```bash
git clone <repository-url>
cd pic_gen_tool_v2
```

#### 2. 安装后端依赖
```bash
cd backend
pip install -r requirements.txt
```

#### 3. 安装前端依赖
```bash
cd ../frontend
pip install -r requirements.txt
```

#### 4. 启动应用

**Windows用户（推荐）**

一键启动（首次使用）：
```bash
start.bat
```
- 自动检查并安装依赖
- 启动后端和前端服务
- 适合首次使用

快速启动（日常使用）：
```bash
quick_start.bat
```
- 跳过依赖检查，快速启动
- 适合日常开发使用

停止服务：
```bash
stop.bat
```

**跨平台启动**

方式一：Python启动脚本
```bash
# 回到项目根目录
cd ..
python start.py
```

方式二：分别启动
```bash
# 启动后端服务
python start.py --mode backend

# 另开终端启动前端
python start.py --mode frontend
```

方式三：手动启动
```bash
# 启动后端
python scripts/start_backend.py

# 启动前端
python scripts/start_frontend.py
```

### 使用说明

#### AI生成功能
1. 选择军事目标（坦克/战机/舰艇）
2. 选择天气条件（雨天/雪天/大雾/夜间）
3. 选择场景环境（城市/岛屿/乡村）
4. 点击"生成提示词"自动构建提示词
5. 调整生成参数（采样步数、CFG Scale等）
6. 点击"生成图像"开始AI生成

#### 传统生成功能
1. 选择军事目标类型
2. 选择天气和场景条件
3. 点击"开始生成"进行传统蒙版合成

### API文档
启动后端服务后，访问 http://localhost:8000/docs 查看完整的API文档。 