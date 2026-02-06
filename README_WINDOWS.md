# Windows 快速启动指南

此目录已包含前端页面、后端服务与文档，适合在 Windows 上直接运行。

## 1. 准备环境
- 安装 Python 3.10+。
- 安装 MySQL/MariaDB，并创建数据库。
- 若后端依赖安装失败（如 `av`），建议使用 Conda 预装：
  - `conda install -c conda-forge av ffmpeg`

## 2. 初始化数据库
在 MySQL 中执行：
1. `init.sql`
2. `schema_additions.sql`（新增 items 与 item_assets 表）

## 3. 启动后端服务
打开 PowerShell，进入后端目录：
```powershell
cd .\project\backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

$env:DATABASE_URL="mysql+pymysql://root:password@127.0.0.1:3306/lerobot_dataset"
$env:DATA_ROOT="C:\path\to\lerobot_package\project\data"
$env:JWT_SECRET="dev-secret"
$env:ADMIN_USERNAME="admin"
$env:ADMIN_PASSWORD="admin"
$env:CORS_ORIGINS="*"
$env:PRIMARY_CAMERA_KEY="top"

python -m uvicorn app.main:app --reload --port 8000
```

## 4. 打开前端
方式 A（直接打开）：双击 `project\111.html`。

方式 B（本地静态服务）：
```powershell
cd .\project
python -m http.server 5090
```
浏览器访问：`http://localhost:5090/111.html`

## 5. 使用流程
1. 登录（默认账号：admin/admin）。
2. 在左侧 “本地数据集导入” 输入数据集路径（例如 `C:\data\lerobot\my_dataset`）。
3. 点击 “解析数据集”，确认返回信息。
4. 输入名称后点击 “创建并索引”，等待索引进度条完成。
5. 选择数据集，进入 episode 列表，点击即可播放。



开启前端：python -m http.server 5090 2>&1 &
开启后端：python start_backend.py 2>&1 &