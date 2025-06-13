# 作战环境下军事目标数据集生成与管理平台

本项目旨在根据《需求规格说明书》和《项目架构与开发计划》构建一个功能完善的数据集生成与管理平台。

## 项目结构概览

```
.
├── backend/                # 后端服务器 (FastAPI, Celery)
│   ├── app/                # 主应用包
│   │   ├── api/            # API 接口 (Routers)
│   │   │   └── v1/
│   │   ├── core/           # 核心逻辑、配置、Celery实例
│   │   ├── crud/           # 数据库交互函数 (Create, Read, Update, Delete)
│   │   ├── db/             # SQLAlchemy 模型和会话管理
│   │   ├── schemas/        # Pydantic 数据模型
│   │   ├── services/       # 业务逻辑服务
│   │   └── tasks/          # Celery 任务定义
│   ├── tests/              # 后端测试
│   └── requirements.txt    # 后端Python依赖
├── data/                   # 生成的数据和原始素材
│   ├── datasets/           # 存储生成的数据集
│   └── raw_assets/         # 存储甲方提供的原始图片素材
├── docs/                   # 项目文档
│   ├── 需求文档.md
│   └── 项目架构与开发计划.md
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

## 快速开始

(此处后续填写环境配置、项目启动等说明) 