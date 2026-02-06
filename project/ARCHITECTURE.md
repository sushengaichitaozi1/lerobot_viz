# LeRobot 数据集管理系统架构文档 v2.0

## 项目概述

这是一个基于 Web 的 LeRobot 机器人数据集管理平台，用于索引、浏览和实时预览机器人训练数据集（Episodes）。系统支持按机器人和任务类型分类管理，提供实时视频切片播放功能。

## 技术栈

### 后端 (Backend)
- **框架**: FastAPI (Python)
- **数据库**: SQLite
- **ORM**: SQLAlchemy
- **认证**: JWT (JSON Web Token)
- **数据处理**: LeRobot 库 (Hugging Face)
- **视频处理**: FFmpeg (实时切片), OpenCV, NumPy

### 前端 (Frontend)
- **框架**: Vue 3
- **样式**: Tailwind CSS
- **图标**: Font Awesome
- **字体**: Manrope, Space Grotesk
- **视频播放**: HTML5 Video Player

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              前端 (Vue 3)                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ 登录认证 │  │ 分类筛选 │  │ 全量展示 │  │ 实时播放 │  │ 详情面板 │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
│                                       │                    │                     │
│                                       └────────────────────┴─────────────────┐   │
│                                                           ↓                   │
│                                              ┌──────────────────────────┐      │
│                                              │   Rerun 可视化按钮        │      │
│                                              │   [生成.rrd] [启动服务]  │      │
│                                              └──────────────────────────┘      │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │ HTTP/REST API
┌───────────────────────────────────────┴─────────────────────────────────────────┐
│                          后端 (FastAPI)                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ JWT认证  │  │ 分类查询 │  │ 实时切片 │  │ 视频流API │  │ 元数据   │         │
│  │ 中间件   │  │ 筛选 API │  │ 服务     │  │ Range请求 │  │ 解析服务 │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │              Rerun 可视化服务 (新增)                                       │  │
│  │  ┌────────────────────┐  ┌────────────────────┐                         │  │
│  │  │   .rrd 文件生成    │  │  WebSocket 服务器  │                         │  │
│  │  │   (供下载)         │  │  (实时可视化)       │                         │  │
│  │  └────────────────────┘  └────────────────────┘                         │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
┌───────────────────────────────────────┴─────────────────────────────────────────┐
│                    SQLite 数据库 + LeRobot 原始文件系统                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                        │
│  │ robots   │  │  types   │  │  items   │  │  tasks   │                        │
│  │(机器人)  │  │(任务类型) │  │(Episodes)│  │(ETL任务) │                        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                        │
└───────────────────────────────────────────────────────────────────────────────────┘
                                        │
┌───────────────────────────────────────┴─────────────────────────────────────────┐
│                    LeRobot 原始数据文件 (只读，不预生成视频)                      │
│  /data/robots/{robot_name}/tasks/{task_type}/episodes/{episode_id}/             │
│     ├── info.json                    # Episode 元数据                            │
│     ├── observation_images/          # 原始图像帧序列                            │
│     │   ├── 000000.png                                                        │
│     │   ├── 000001.png                                                        │
│     │   └── ...                                                               │
│     └── observation_images/*/       # 多相机视角 (top, wrist, etc.)            │
└───────────────────────────────────────────────────────────────────────────────────┘
                                        │
┌───────────────────────────────────────┴─────────────────────────────────────────┐
│                         Rerun 可视化输出                                         │
│  /downloads/rrd/                                                                  │
│     ├── so101_episode_000001.rrd    # Rerun 数据文件                            │
│     ├── so101_episode_000002.rrd                                               │
│     └── ...                                                                    │
│                                                                                  │
│  用户本地查看: rerun xxx.rrd                                                     │
│  或浏览器访问: ws://server:9087                                                  │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## 核心设计原则

### 1. 实时切片，不预存储
- **原始原则**: 不再为每个 Episode 预先生成视频文件
- **新方案**: 用户点击播放时，后端实时从图像序列中切片并流式返回
- **优势**:
  - 节省大量存储空间
  - 支持任意分辨率、帧率、编码参数切换
  - 无需维护视频文件同步问题

### 2. 元数据入库，结构化存储
- **解析入库**: 所有 Episode 基本信息在索引时解析并写入数据库
- **存储内容**:
  - 帧数、时长、分辨率、FPS
  - 机器人名称、任务类型
  - 相机视角列表
  - 图像路径、文件大小统计
- **查询优化**: 所有列表查询直接从数据库读取，无需读取文件系统

### 3. 分类体系，多维筛选
- **一级分类**: 机器人名称 (robot_name)
- **二级分类**: 任务类型 (task_type)
- **支持操作**:
  - 按机器人筛选
  - 按任务类型筛选
  - 组合筛选
  - 全部展示

---

## 核心设计要点 (确认版)

### 分类功能实现
界面的分类筛选功能 **完全基于数据库字段**：
- `robot_id` (外键 → Robot.id) - 按机器人分类
- `task_type_id` (外键 → TaskType.id) - 按任务类型分类

**前端交互**:
1. 顶部分类按钮显示所有机器人 (`GET /robots`)
2. 点击机器人后，显示该机器人的任务类型 (`GET /robots/{name}/types`)
3. 调用 `GET /items?robot=xxx&type=xxx` 获取筛选后的 Episodes

### 视频存储策略 (重要)
- **唯一存储位置**: 本地 LeRobot 原始数据目录
- **不预生成视频**: 系统中不存在任何 `.mp4` 视频文件
- **只存储原始图像**: `/data/robots/{robot}/tasks/{type}/episodes/{id}/observation_images/{camera}/*.png`

### 视频播放流程 (每次点击都实时切片)
```
用户点击 Episode 卡片
       ↓
前端请求: GET /items/{id}/stream?camera=top
       ↓
后端从数据库读取:
  - item.file_path (图像序列所在目录)
  - camera_info.image_path (相机图像路径)
  - camera_info.file_pattern (文件命名模式)
       ↓
后端启动 FFmpeg 进程 (实时)
  输入: 本地图像序列 (如 /data/.../observation_images/top/%06d.png)
  处理: 编码为 H.264 MP4
  输出: pipe:1 (内存管道，不落地)
       ↓
流式返回视频数据到前端
```

**关键点**: 每次用户点击播放，都会启动一个新的 FFmpeg 进程实时切片视频。

### 数据流向图
```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户界面 (前端)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ 分类按钮    │  │ Episode列表 │  │ 视频播放器  │                │
│  │ (robot_id)  │  │ (来自DB)    │  │ (实时切片)  │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HTTP API
┌───────────────────────────┴─────────────────────────────────────────┐
│                         后端服务                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ 筛选查询    │  │ 元数据读取  │  │ FFmpeg切片  │                │
│  │ (DB查询)    │  │ (DB查询)    │  │ (实时处理)  │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────────┐
│                      数据存储层                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    SQLite 数据库                             │   │
│  │  - robots (id, name, display_name)                          │   │
│  │  - task_types (id, name, robot_id, display_name)            │   │
│  │  - items (id, robot_id, task_type_id, file_path, ...)       │   │
│  │  - camera_infos (id, item_id, camera_key, image_path, ...)  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              本地文件系统 (只读)                             │   │
│  │  /data/robots/{robot}/tasks/{type}/episodes/{id}/           │   │
│  │      └── observation_images/{camera}/*.png  (原始图像)      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 核心功能模块

### 1. 分类管理模块

#### 后端 API
- **GET /robots** - 获取所有机器人列表
- **GET /robots/{name}** - 获取单个机器人详情
- **GET /robots/{name}/types** - 获取机器人的所有任务类型
- **GET /types** - 获取所有任务类型列表
- **GET /types/{name}/robots** - 获取该任务类型下的所有机器人

#### 数据库模型 (Robot)
```python
class Robot(Base):
    id: 主键
    name: 机器人名称 (唯一)
    display_name: 显示名称
    description: 描述
    created_at: 创建时间
```

#### 数据库模型 (TaskType)
```python
class TaskType(Base):
    id: 主键
    name: 任务类型名称 (如 "pick_and_place", "sorting")
    display_name: 显示名称
    robot_id: 所属机器人 (外键 → Robot.id)
    description: 描述
    created_at: 创建时间
```

### 2. Episode 管理模块

#### 后端 API
- **GET /items** - 获取所有 Episodes（支持分页和筛选）
  - Query 参数: `robot`, `type`, `page`, `page_size`, `search`
- **GET /items/{id}** - 获取单个 Episode 详情
- **GET /items/{id}/cameras** - 获取 Episode 的所有相机视角列表
- **DELETE /items/{id}** - 软删除 Episode
- **POST /items/index** - 手动触发索引单个 Episode

#### 数据库模型 (Item) - 扩展版
```python
class Item(Base):
    # 基础字段
    id: 主键
    episode_id: Episode ID (来自 LeRobot)

    # 分类关联
    robot_id: 所属机器人 (外键 → Robot.id)
    task_type_id: 任务类型 (外键 → TaskType.id)

    # 文件路径
    file_path: Episode 根目录路径

    # 视频元数据 (解析后入库)
    total_frames: 总帧数
    fps: 帧率
    duration_s: 时长 (秒)
    width: 图像宽度
    height: 图像高度

    # 统计信息
    image_count: 图像文件总数
    total_size_bytes: 总文件大小
    camera_count: 相机视角数量

    # 时间戳
    created_at: 创建时间
    updated_at: 更新时间
    indexed_at: 索引时间
    deleted_at: 软删除时间
```

#### 数据库模型 (CameraInfo)
```python
class CameraInfo(Base):
    id: 主键
    item_id: 所属 Episode (外键 → Item.id)
    camera_key: 相机标识 (如 "top", "wrist")
    display_name: 显示名称
    image_path: 图像序列路径
    frame_count: 该相机的帧数
    width: 该相机的图像宽度
    height: 该相机的图像高度
```

### 3. 实时视频切片服务模块

#### 后端 API
- **GET /items/{id}/stream** - 获取 Episode 的实时切片视频流
  - Query 参数: `camera` (相机视角, 默认: top)
  - 支持标准 HTTP Range 请求，支持拖拽播放
  - 响应格式: video/mp4

- **GET /items/{id}/thumbnail** - 获取 Episode 缩略图
  - Query 参数: `camera`, `frame` (指定帧号)

#### 视频切片处理流程

```
用户点击播放视频
       ↓
前端请求: GET /items/{id}/stream?camera=top
       ↓
后端接收请求
       ↓
从数据库读取 Episode 元数据 (路径、帧数、FPS等)
       ↓
读取原始图像序列:
  /data/robots/{robot}/tasks/{type}/episodes/{id}/observation_images/{camera}/*.png
       ↓
FFmpeg 管道处理 (实时):
  输入: PNG 图像序列
  ↓
  处理: 调整分辨率、设置FPS、编码 (H.264)
  ↓
  输出: MP4 视频流 (内存中，不落地)
       ↓
流式响应到前端 (支持 Range 分段请求)
       ↓
HTML5 Video 播放器接收并播放
```

#### 视频处理参数 (可配置)
```python
VIDEO_ENCODING_CONFIG = {
    "codec": "libx264",           # H.264 编码
    "preset": "fast",             # 编码速度预设
    "crf": 23,                    # 质量控制 (18-28)
    "pix_fmt": "yuv420p",         # 像素格式
    "movflags": "faststart",      # 快速启动 (优化流式播放)
    "tune": "zerolatency",        # 零延迟调优
}
```

#### 性能优化策略
- **缓存策略**: 可选的短期内存缓存 (LRU, 缓存最近 N 个切片)
- **预加载**: 前端可请求预加载接下来的几秒
- **并发控制**: 限制同时切片的任务数，防止资源耗尽
- **分辨率档位**: 提供 360p/480p/720p/1080p 多档位选择

### 4. 元数据解析服务模块

#### 解析流程

```
LeRobot 数据集路径扫描
       ↓
识别机器人目录 → 创建/更新 Robot 记录
       ↓
识别任务类型目录 → 创建/更新 TaskType 记录
       ↓
扫描 Episode 目录
       ↓
┌─────────────────────────────────────────┐
│  对每个 Episode:                         │
│  1. 读取 info.json 或 meta.json          │
│  2. 扫描 observation_images/ 目录         │
│  3. 识别所有相机视角 (top, wrist, etc.)   │
│  4. 统计每个相机的帧数、分辨率             │
│  5. 计算总帧数、时长、FPS                  │
│  6. 计算文件大小                          │
│  7. 写入数据库 (Item + CameraInfo)        │
└─────────────────────────────────────────┘
       ↓
完成索引，生成统计报告
```

#### 后端 API
- **POST /index/scan** - 扫描指定路径并索引
  - Body: `{ "path": "D:/lerobot_data" }`
- **POST /index/rebuild` - 重建索引 (清空后重新扫描)
- **GET /index/status` - 获取索引状态和进度
- **POST /index/robot/{robot_name}` - 索引单个机器人

### 5. Rerun 数据可视化模块 (新增)

使用 **Rerun** 进行 Episode 全数据可视化，包括：
- 多相机图像时间轴
- 动作空间 (action) 各维度曲线
- 观测状态 (state) 各维度曲线
- 奖励 (reward) 曲线
- done 标志时间点

#### 后端 API
- **POST /items/{id}/visualize/rrd** - 生成 .rrd 文件并返回下载链接
  - Body: `{ "format": "rrd" }`
  - 响应: `{ "download_url": "/downloads/xxx.rrd", "file_path": "..." }`

- **POST /items/{id}/visualize/server** - 启动 Rerun 远程服务器
  - Body: `{ "mode": "distant", "ws_port": 9087 }`
  - 响应: `{ "ws_url": "ws://server:9087", "web_url": "http://server:9090" }`

- **GET /downloads/{filename}** - 下载生成的 .rrd 文件

#### Rerun 可视化服务
```python
# services/rerun_viz.py
import rerun as rr
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def generate_rrd(item_id: int, output_dir: Path) -> Path:
    """生成 .rrd 文件供下载"""
    # 1. 从数据库读取 Episode 信息
    item = get_item(item_id)

    # 2. 加载 LeRobot 数据集
    dataset = LeRobotDataset(
        repo_id=item.robot.name,
        root=item.file_path.parent.parent.parent,  # 回溯到数据集根目录
        episodes=[int(item.episode_id.split("_")[-1])]
    )

    # 3. 使用 Rerun 记录数据
    rr.init(f"{item.robot.name}_episode_{item.episode_id}", spawn=False)

    for batch in dataset:
        # 记录图像、动作、状态等
        for key in dataset.meta.camera_keys:
            rr.log(key, rr.Image(batch[key]))
        # ... 记录其他数据

    # 4. 保存为 .rrd 文件
    rrd_path = output_dir / f"{item.robot.name}_{item.episode_id}.rrd"
    rr.save(rrd_path)
    return rrd_path
```

#### 前端集成
在详情面板中添加 "Rerun 可视化" 按钮：
```
┌─────────────────────────────────────────┐
│  Episode: episode_000001               │
│  ─────────────────────────────────────  │
│                                          │
│  [ 播放视频 ]  [ Rerun 可视化 ]          │  ← 新增按钮
│                                          │
│  Rerun 可视化包含:                       │
│  • 多相机图像时间轴                      │
│  • 动作/状态曲线图                       │
│  • 奖励/done 标志                        │
│                                          │
│  [ 生成 .rrd 文件 ]  [ 启动服务器 ]      │
└─────────────────────────────────────────┘
```

#### 用户使用流程

**方式1: 生成 .rrd 文件下载**
```
用户点击 "生成 .rrd 文件"
       ↓
POST /items/{id}/visualize/rrd
       ↓
后端生成 .rrd 文件 (约 10-30 秒)
       ↓
返回下载链接
       ↓
用户下载文件，本地运行: rerun xxx.rrd
```

**方式2: 启动 Rerun 服务器**
```
用户点击 "启动服务器"
       ↓
POST /items/{id}/visualize/server
       ↓
后端启动 Rerun WebSocket 服务器
       ↓
返回 ws://server:9087
       ↓
用户在浏览器或 rerun 客户端中连接
```

#### 技术说明
- **依赖**: `pip install rerun torch numpy tqdm lerobot`
- **.rrd 文件**: Rerun 专用的二进制格式，包含完整的时间序列数据
- **优势**:
  - 无需预生成视频，随时可视化
  - 支持多路数据同步展示
  - 可交互式时间轴浏览
  - 支持远程服务器模式

### 6. 用户认证模块 (保持不变)

#### 后端 API
- **POST /auth/login** - 用户登录
- **POST /auth/refresh** - 刷新 token
- **GET /auth/me** - 获取当前用户信息

### 7. 前端页面功能

#### 主界面布局 (全新设计)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  Header                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐  │
│  │  Logo + 标题     │  │    搜索框        │  │  API状态 | 用户信息 | 登出   │  │
│  └──────────────────┘  └──────────────────┘  └─────────────────────────────┘  │
├────────────────────────────────────────────────────────────────────────────────┤
│  顶部分类筛选栏                                                                 │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────────────────────┐   │
│  │ 全部 │ │ 机器人A │ │ 机器人B │ │ 机器人C │ │ 更多... │ ▼              │   │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ │                          │   │
│                                                  └──────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ 子类型筛选 (动态显示)                                                     │  │
│  │ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                                    │  │
│  │ │ 全部 │ │ pick  │ │ place │ │ sort │ ...                              │  │
│  │ └──────┘ └──────┘ └──────┘ └──────┘                                    │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────────────────────┤
│  统计信息栏                                                                     │
│  共 1,234 个 Episodes | 已选: 机器人A - pick_and_place | 显示: 1-50 / 156      │
├───────────────────────────────┬────────────────────────────────────────────────┤
│                               │                                                │
│  Episode 网格展示区           │   详情面板 (右侧滑出/弹窗)                      │
│  (全量展示，分页加载)          │                                                │
│                               │   ┌─────────────────────────────────────────┐ │
│  ┌──────┐ ┌──────┐ ┌──────┐   │   │  Episode: episode_000001               │ │
│  │ EP1  │ │ EP2  │ │ EP3  │   │   │  机器人: RobotA | 类型: pick           │ │
│  │缩略图│ │缩略图│ │缩略图│   │   │  ─────────────────────────────────────  │ │
│  └──────┘ └──────┘ └──────┘   │   │                                          │ │
│                               │   │  视频播放器                                │ │
│  ┌──────┐ ┌──────┐ ┌──────┐   │   │  ┌────────────────────────────────────┐ │ │
│  │ EP4  │ │ EP5  │ │ EP6  │   │   │  │         ▶  播放控制栏               │ │ │
│  │缩略图│ │缩略图│ │缩略图│   │   │  └────────────────────────────────────┘ │ │
│  └──────┘ └──────┘ └──────┘   │   │                                          │ │
│                               │   │  相机视角选择:                             │ │
│  ┌──────┐ ┌──────┐ ┌──────┐   │   │  [Top] [Wrist] [Side1] [Side2]          │ │
│  │ EP7  │ │ EP8  │ │ EP9  │   │   │                                          │ │
│  └──────┘ └──────┘ └──────┘   │   │  基本信息:                                │ │
│                               │   │  • 帧数: 1500                              │ │
│         ... (更多 Episodes)    │   │  • 时长: 50s @ 30fps                      │ │
│                               │   │  • 分辨率: 640x480                         │ │
│                               │   │  • 大小: 125 MB                            │ │
│                               │   │                                          │ │
│                               │   │  [ 下载 ] [ 删除 ]                         │ │
│                               │   └─────────────────────────────────────────┘ │ │
│                               │                                                │
└───────────────────────────────┴────────────────────────────────────────────────┘
│  分页控件                                                                         │
│  [< 上一页] 第 1/25 页 [下一页 >]  跳转: [___]                                   │
└────────────────────────────────────────────────────────────────────────────────┘
```

#### 分类筛选交互 (基于 robot_id 和 task_type_id)

**一级分类: 按机器人筛选 (robot_id)**
```
步骤1: 加载所有机器人
  GET /robots
  Response: [{id: 1, name: "so101", display_name: "SO-101"}, ...]

步骤2: 点击机器人按钮
  GET /items?robot=so101&page=1&page_size=50
  SQL: SELECT * FROM items WHERE robot_id = (SELECT id FROM robots WHERE name = 'so101')
```

**二级分类: 按任务类型筛选 (task_type_id)**
```
步骤1: 选择机器人后，加载该机器人的任务类型
  GET /robots/so101/types
  Response: [{id: 1, name: "pick_and_place"}, {id: 2, name: "sorting"}, ...]

步骤2: 点击任务类型按钮
  GET /items?robot=so101&type=pick_and_place&page=1&page_size=50
  SQL: SELECT * FROM items
       WHERE robot_id = (SELECT id FROM robots WHERE name = 'so101')
       AND task_type_id = (SELECT id FROM task_types WHERE name = 'pick_and_place')
```

**组合筛选流程**:
1. 页面加载时: 调用 `GET /robots` 显示所有机器人按钮
2. 点击"机器人A": 调用 `GET /items?robot=机器人A` 并调用 `GET /robots/机器人A/types` 显示任务类型
3. 点击任务类型"pick": 调用 `GET /items?robot=机器人A&type=pick`
4. 点击"全部": 调用 `GET /items` (不带筛选参数)

**前端状态管理**:
```javascript
// Vue 组件状态
const filters = ref({
  robot: null,      // 当前选中的 robot_id 或 robot name
  type: null,       // 当前选中的 task_type_id 或 type name
  page: 1,
  pageSize: 50
});

// 加载 Episodes
async function loadItems() {
  const params = new URLSearchParams();
  if (filters.value.robot) params.append('robot', filters.value.robot);
  if (filters.value.type) params.append('type', filters.value.type);
  params.append('page', filters.value.page);
  params.append('page_size', filters.value.pageSize);

  const response = await fetch(`/items?${params}`);
  // ...
}
```

#### Episode 卡片信息
- 缩略图 (默认 top 相机第一帧)
- Episode ID (如 episode_000001)
- 机器人名称
- 任务类型
- 时长信息
- 悬停时显示更多信息

#### 视频播放交互
1. **点击卡片**: 右侧滑出详情面板
2. **自动加载**: 自动加载并播放默认相机 (top) 的视频
3. **实时切片**: 视频通过 API 实时切片返回
4. **相机切换**: 点击相机按钮，实时切换到对应相机的视频
5. **拖拽播放**: 支持视频进度条拖拽，触发 Range 请求

## 数据关系图 (分类机制)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          分类查询原理                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   robots 表                          task_types 表                              │
│  ┌─────────────────────┐          ┌─────────────────────────────┐              │
│  │ id │ name      │ ...│          │ id │ name  │ robot_id │ ... │              │
│  ├─────────────────────┤          ├─────────────────────────────┤              │
│  │ 1  │ so101     │    │◄─────────│ 1  │ pick   │ 1        │     │              │
│  │ 2  │ so102     │    │  外键    │ 2  │ place  │ 1        │     │              │
│  │ 3  │ aloha     │    │          │ 3  │ sort   │ 1        │     │              │
│  └─────────────────────┘          │ 4  │ pick   │ 2        │     │              │
│                                    │ 5  │ place  │ 3        │     │              │
│                                    └─────────────────────────────┘              │
│                                                │                                 │
│                                                │ 外键                            │
│                                                ▼                                 │
│   items 表                                                                     │
│  ┌───────────────────────────────────────────────────────────────────┐         │
│  │ id │ robot_id │ task_type_id │ episode_id │ file_path       │ ... │         │
│  ├───────────────────────────────────────────────────────────────────┤         │
│  │ 1  │ 1        │ 1            │ ep_00001   │ /data/so101/... │     │         │
│  │ 2  │ 1        │ 1            │ ep_00002   │ /data/so101/... │     │         │
│  │ 3  │ 1        │ 2            │ ep_00003   │ /data/so101/... │     │         │
│  │ 4  │ 2        │ 4            │ ep_00001   │ /data/so102/... │     │         │
│  │ 5  │ 3        │ 5            │ ep_00001   │ /data/aloha/... │     │         │
│  └───────────────────────────────────────────────────────────────────┘         │
│                                                                                 │
│   筛选示例:                                                                     │
│   ───────────────────────────────────────────────────────────────────────────  │
│   GET /items?robot=so101&type=pick                                             │
│   → SELECT * FROM items                                                        │
│     WHERE robot_id = 1           -- (so101 的 id)                              │
│     AND task_type_id = 1         -- (pick 的 id)                               │
│     → 返回: id=1, id=2 的两条记录                                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 数据库 Schema v2.0

```sql
-- 机器人表
CREATE TABLE robots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) UNIQUE NOT NULL,           -- 机器人唯一标识
    display_name VARCHAR(255),                   -- 显示名称
    description TEXT,                            -- 描述
    metadata JSON,                               -- 额外元数据 (JSON)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 任务类型表
CREATE TABLE task_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,                  -- 任务类型名称
    display_name VARCHAR(255),                   -- 显示名称
    robot_id INTEGER NOT NULL,                   -- 所属机器人
    description TEXT,                            -- 描述
    metadata JSON,                               -- 额外元数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (robot_id) REFERENCES robots(id) ON DELETE CASCADE,
    UNIQUE(robot_id, name)                       -- 同一机器人下类型名唯一
);

-- Episodes 表 (扩展版)
CREATE TABLE items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id VARCHAR(255),                     -- LeRobot Episode ID

    -- 分类关联
    robot_id INTEGER NOT NULL,                   -- 所属机器人
    task_type_id INTEGER NOT NULL,               -- 任务类型

    -- 文件路径
    file_path VARCHAR(1024) NOT NULL,            -- Episode 根目录

    -- 视频元数据 (解析后入库)
    total_frames INTEGER,                        -- 总帧数
    fps FLOAT,                                   -- 帧率
    duration_s FLOAT,                            -- 时长 (秒)
    width INTEGER,                               -- 图像宽度
    height INTEGER,                              -- 图像高度

    -- 统计信息
    image_count INTEGER,                         -- 图像文件总数
    total_size_bytes BIGINT,                     -- 总文件大小
    camera_count INTEGER,                        -- 相机视角数量

    -- 索引状态
    index_status VARCHAR(50) DEFAULT 'pending',  -- 索引状态: pending/indexing/completed/failed
    index_error TEXT,                            -- 索引错误信息

    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    indexed_at TIMESTAMP,                        -- 最后索引时间
    deleted_at TIMESTAMP,                        -- 软删除时间

    FOREIGN KEY (robot_id) REFERENCES robots(id) ON DELETE CASCADE,
    FOREIGN KEY (task_type_id) REFERENCES task_types(id) ON DELETE CASCADE
);

-- 相机信息表
CREATE TABLE camera_infos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL,                    -- 所属 Episode
    camera_key VARCHAR(100) NOT NULL,            -- 相机标识 (top, wrist, etc.)
    display_name VARCHAR(255),                   -- 显示名称
    image_path VARCHAR(1024) NOT NULL,           -- 图像序列路径
    frame_count INTEGER,                         -- 该相机的帧数
    width INTEGER,                               -- 图像宽度
    height INTEGER,                              -- 图像高度
    file_pattern VARCHAR(255),                   -- 文件命名模式 (如 %06d.png)

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE CASCADE,
    UNIQUE(item_id, camera_key)                  -- 同一 Episode 下相机 key 唯一
);

-- 索引任务表 (可选，用于跟踪索引进度)
CREATE TABLE index_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type VARCHAR(50) NOT NULL,              -- scan, rebuild, single
    target_path VARCHAR(1024),                   -- 目标路径
    status VARCHAR(50) DEFAULT 'pending',        -- pending/running/completed/failed
    progress INTEGER DEFAULT 0,                  -- 进度 0-100
    total_episodes INTEGER,                      -- 总 Episode 数
    processed_episodes INTEGER DEFAULT 0,        -- 已处理数
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引起始日志表 (可选)
CREATE TABLE index_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER,
    level VARCHAR(20),                           -- info/warning/error
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES index_tasks(id) ON DELETE CASCADE
);

-- 用户表 (认证用)
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引以优化查询
CREATE INDEX idx_items_robot ON items(robot_id);
CREATE INDEX idx_items_task_type ON items(task_type_id);
CREATE INDEX idx_items_robot_type ON items(robot_id, task_type_id);
CREATE INDEX idx_items_deleted ON items(deleted_at) WHERE deleted_at IS NULL;
CREATE INDEX idx_cameras_item ON camera_infos(item_id);
```

## API 请求示例 v2.0

### 获取机器人列表
```bash
GET /robots

Response:
{
  "robots": [
    {
      "id": 1,
      "name": "so101",
      "displayName": "SO-101 Robot",
      "description": "Standard pick and place robot",
      "typeCount": 3,
      "episodeCount": 156
    }
  ]
}
```

### 获取机器人的任务类型
```bash
GET /robots/so101/types

Response:
{
  "taskTypes": [
    {
      "id": 1,
      "name": "pick_and_place",
      "displayName": "Pick and Place",
      "episodeCount": 120
    },
    {
      "id": 2,
      "name": "sorting",
      "displayName": "Sorting",
      "episodeCount": 36
    }
  ]
}
```

### 获取 Episodes (带筛选)
```bash
GET /items?robot=so101&type=pick_and_place&page=1&page_size=50

Response:
{
  "items": [
    {
      "id": 1,
      "episodeId": "episode_000001",
      "robot": "so101",
      "taskType": "pick_and_place",
      "thumbnailUrl": "/items/1/thumbnail?camera=top",
      "totalFrames": 1500,
      "duration": 50.0,
      "fps": 30.0,
      "resolution": "640x480",
      "cameraCount": 4
    },
    ...
  ],
  "total": 120,
  "page": 1,
  "pageSize": 50
}
```

### 获取 Episode 详情
```bash
GET /items/1

Response:
{
  "id": 1,
  "episodeId": "episode_000001",
  "robot": {
    "id": 1,
    "name": "so101",
    "displayName": "SO-101 Robot"
  },
  "taskType": {
    "id": 1,
    "name": "pick_and_place",
    "displayName": "Pick and Place"
  },
  "filePath": "/data/robots/so101/tasks/pick_and_place/episodes/episode_000001",
  "totalFrames": 1500,
  "fps": 30.0,
  "duration": 50.0,
  "width": 640,
  "height": 480,
  "imageCount": 6000,
  "totalSizeBytes": 131072000,
  "cameraCount": 4,
  "cameras": [
    {
      "cameraKey": "top",
      "displayName": "Top Camera",
      "frameCount": 1500,
      "width": 640,
      "height": 480
    },
    {
      "cameraKey": "wrist",
      "displayName": "Wrist Camera",
      "frameCount": 1500,
      "width": 480,
      "height": 640
    }
  ]
}
```

### 获取视频流 (实时切片)
```bash
GET /items/1/stream?camera=top

Headers:
  Range: bytes=0-1024000  (支持分段请求)

Response:
Content-Type: video/mp4
Accept-Ranges: bytes
Content-Length: 1024000
Content-Range: bytes 0-1024000/10485760

[二进制视频流数据...]
```

### 获取缩略图
```bash
GET /items/1/thumbnail?camera=top&frame=0

Response:
Content-Type: image/png

[二进制图像数据...]
```

### 扫描并索引数据集
```bash
POST /index/scan
Content-Type: application/json

{
  "path": "D:/lerobot_data"
}

Response:
{
  "taskId": 42,
  "status": "running",
  "message": "Index task started"
}
```

### 查询索引状态
```bash
GET /index/status?task_id=42

Response:
{
  "taskId": 42,
  "status": "running",
  "progress": 65,
  "totalEpisodes": 200,
  "processedEpisodes": 130,
  "currentPath": "so101/tasks/pick_and_place/episodes/episode_000131"
}
```

### 生成 Rerun .rrd 文件
```bash
POST /items/1/visualize/rrd
Content-Type: application/json

{
  "format": "rrd"
}

Response:
{
  "status": "completed",
  "downloadUrl": "/downloads/rrd/so101_episode_000001.rrd",
  "filePath": "/downloads/rrd/so101_episode_000001.rrd",
  "fileSize": 15728640,
  "duration": 50.0
}

# 下载文件
GET /downloads/rrd/so101_episode_000001.rrd
```

### 启动 Rerun 可视化服务器
```bash
POST /items/1/visualize/server
Content-Type: application/json

{
  "mode": "distant",
  "ws_port": 9087,
  "web_port": 9090
}

Response:
{
  "status": "running",
  "wsUrl": "ws://localhost:9087",
  "webUrl": "http://localhost:9090",
  "instructions": "使用 rerun ws://localhost:9087 或访问 http://localhost:9090"
}
```

## 配置说明 v2.0

### 后端配置 (config.py)
```python
# API 配置
api_title = "LeRobot Dataset API v2.0"

# 数据库配置
database_url = "sqlite:///lerobot_dataset_v2.db"

# 数据根目录
data_root = "D:/lerobot_package/project/data"

# CORS 配置
cors_origins = ["http://localhost:5090"]

# JWT 配置
jwt_secret = "your-secret-key"
jwt_expire_seconds = 86400  # 24 hours

# 视频处理配置
video_encoding = {
    "codec": "libx264",
    "preset": "fast",
    "crf": 23,
    "pix_fmt": "yuv420p",
    "movflags": "faststart",
    "tune": "zerolatency",
    "g": 30,              # GOP size
    "bframes": 0,         # B-frames (流式播放设为0)
}

# 视频流配置
stream_config = {
    "max_concurrent_streams": 5,      # 最大并发切片数
    "cache_enabled": True,            # 是否启用缓存
    "cache_size": 10,                 # 缓存数量
    "cache_ttl": 300,                 # 缓存过期时间 (秒)
}

# 分辨率档位
resolutions = {
    "360p": {"width": 480, "height": 360},
    "480p": {"width": 640, "height": 480},
    "720p": {"width": 1280, "height": 720},
    "1080p": {"width": 1920, "height": 1080},
}

# 默认相机
primary_camera_key = "top"

# 分页配置
default_page_size = 50
max_page_size = 200

# Rerun 可视化配置
rerun_config = {
    "downloads_dir": "downloads/rrd",       # .rrd 文件输出目录
    "ws_port": 9087,                        # WebSocket 端口
    "web_port": 9090,                       # Web 界面端口
    "auto_cleanup_days": 7,                 # 自动清理超过 N 天的 .rrd 文件
    "batch_size": 32,                       # 数据加载批次大小
    "num_workers": 4,                       # 数据加载进程数
}
```

## 文件结构 v2.0

```
lerobot_package/
├── project/
│   ├── backend/
│   │   └── app/
│   │       ├── main.py              # FastAPI 主应用
│   │       ├── config.py            # 配置管理
│   │       ├── db.py                # 数据库连接
│   │       ├── models.py            # SQLAlchemy 模型
│   │       ├── schemas.py           # Pydantic 模型
│   │       ├── auth.py              # JWT 认证
│   │       │
│   │       ├── services/
│   │       │   ├── indexer.py       # 元数据解析服务
│   │       │   ├── video_streamer.py # 实时视频切片服务
│   │       │   ├── thumbnail.py     # 缩略图生成服务
│   │       │   └── rerun_viz.py     # Rerun 可视化服务 (新增)
│   │       │
│   │       ├── api/
│   │       │   ├── auth.py          # 认证 API
│   │       │   ├── robots.py        # 机器人 API
│   │       │   ├── items.py         # Episodes API
│   │       │   ├── stream.py        # 视频流 API
│   │       │   ├── visualize.py     # Rerun 可视化 API (新增)
│   │       │   └── index.py         # 索引 API
│   │       │
│   │       └── utils/
│   │           ├── ffmpy.py         # FFmpeg 封装
│   │           └── cache.py         # 缓存管理
│   │
│   ├── data/                        # LeRobot 原始数据 (只读)
│   │   └── robots/
│   │       └── {robot_name}/
│   │           └── tasks/
│   │               └── {task_type}/
│   │                   └── episodes/
│   │                       └── {episode_id}/
│   │                           ├── info.json
│   │                           └── observation_images/
│   │                               └── {camera_key}/
│   │                                   └── *.png
│   │
│   ├── downloads/                   # 下载文件输出目录
│   │   └── rrd/                     # Rerun .rrd 文件
│   │       ├── so101_episode_000001.rrd
│   │       └── ...
│   │
│   ├── static/                      # 静态资源
│   └── 111.html                     # 单页应用入口
│
├── standalone_dataset_viz.py        # 独立 Rerun 可视化脚本 (保留)
├── lerobot_dataset_v2.db            # SQLite 数据库 v2.0
└── ARCHITECTURE.md                  # 本文档
```

## 开发说明

### 启动后端
```bash
cd D:/lerobot_package/project/backend
python start_backend.py
```

### 启动前端
```bash
cd D:/lerobot_package/project
python -m http.server 5090
```

### 访问地址
- 前端: http://localhost:5090/111.html
- 后端 API: http://localhost:8000
- API 文档: http://localhost:8000/docs

### 索引数据集
```bash
# 扫描并索引整个数据目录
curl -X POST http://localhost:8000/index/scan \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"path": "D:/lerobot_data"}'

# 查询索引进度
curl http://localhost:8000/index/status?task_id=1 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 技术实现要点

### 1. 实时视频切片实现

#### FFmpeg 管道处理
```python
# 后端伪代码
def stream_video(item_id: int, camera: str, range_header: str = None):
    # 1. 从数据库读取 Episode 信息
    item = get_item(item_id)
    camera_info = get_camera_info(item_id, camera)

    # 2. 构建图像序列路径
    image_pattern = f"{camera_info.image_path}/{camera_info.file_pattern}"

    # 3. FFmpeg 命令 (输出到 stdout)
    cmd = [
        "ffmpeg",
        "-i", image_pattern,           # 输入图像序列
        "-c:v", "libx264",             # H.264 编码
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "faststart",
        "-tune", "zerolatency",
        "-f", "mp4",                   # MP4 格式
        "pipe:1"                       # 输出到 stdout
    ]

    # 4. 启动 FFmpeg 进程
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    # 5. 流式返回
    def generate():
        while True:
            chunk = process.stdout.read(8192)
            if not chunk:
                break
            yield chunk

    return StreamingResponse(generate(), media_type="video/mp4")
```

#### Range 请求支持
```python
# 处理分段请求
if range_header:
    # 解析 Range: bytes=0-1024000
    start, end = parse_range(range_header)

    # 使用 FFmpeg -ss 和 -t 参数切片指定范围
    cmd.extend([
        "-ss", str(start / item.total_frames * item.duration),  # 开始时间
        "-t", str((end - start) / item.total_frames * item.duration)  # 时长
    ])
```

### 2. 元数据解析实现

```python
# 解析单个 Episode
def parse_episode(episode_path: str, robot: Robot, task_type: TaskType) -> dict:
    # 1. 读取 meta.json 或 info.json
    meta_path = os.path.join(episode_path, "info.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # 2. 扫描相机目录
    cameras_path = os.path.join(episode_path, "observation_images")
    cameras = []
    for camera_key in os.listdir(cameras_path):
        camera_path = os.path.join(cameras_path, camera_key)
        images = sorted(os.listdir(camera_path))
        frame_count = len(images)

        # 读取第一张图片获取分辨率
        first_img = cv2.imread(os.path.join(camera_path, images[0]))
        height, width = first_img.shape[:2]

        cameras.append({
            "camera_key": camera_key,
            "image_path": camera_path,
            "frame_count": frame_count,
            "width": width,
            "height": height
        })

    # 3. 计算总体信息
    primary_camera = cameras[0]  # 假设第一个是主相机
    total_frames = primary_camera["frame_count"]
    fps = meta.get("fps", 30)
    duration = total_frames / fps

    # 4. 计算文件大小
    total_size = sum(
        os.path.getsize(os.path.join(camera_path, f))
        for camera in cameras
        for f in os.listdir(camera["image_path"])
    )

    return {
        "episode_id": os.path.basename(episode_path),
        "robot_id": robot.id,
        "task_type_id": task_type.id,
        "file_path": episode_path,
        "total_frames": total_frames,
        "fps": fps,
        "duration_s": duration,
        "width": primary_camera["width"],
        "height": primary_camera["height"],
        "image_count": sum(c["frame_count"] for c in cameras),
        "total_size_bytes": total_size,
        "camera_count": len(cameras),
        "cameras": cameras
    }
```

### 3. 缓存策略实现

```python
# LRU 缓存
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

class VideoStreamCache:
    def __init__(self, max_size: int = 10, ttl: int = 300):
        self.cache = {}  # {key: (data, timestamp)}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()

    def get(self, key: str):
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return data
                else:
                    del self.cache[key]
        return None

    def set(self, key: str, data):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # 删除最旧的
                oldest = min(self.cache.items(), key=lambda x: x[1][1])
                del self.cache[oldest[0]]
            self.cache[key] = (data, time.time())

# 缓存键
def make_cache_key(item_id: int, camera: str, resolution: str) -> str:
    return f"{item_id}:{camera}:{resolution}"
```

## 迁移指南 v1.0 → v2.0

### 数据库迁移
1. 备份现有数据库
2. 创建新表 (robots, task_types)
3. 扩展 items 表 (添加新字段)
4. 创建 camera_infos 表
5. 重新索引数据集以填充新字段

### API 变更
1. 新增 `/robots`, `/types` 端点
2. `/items` 端点新增筛选参数
3. `/items/{id}/stream` 替代原有的预览 API
4. 删除 `/items/{id}/preview` 中的视频 URL

### 前端变更
1. 添加顶部分类筛选栏
2. 修改详情面板，移除预生成视频逻辑
3. 视频播放器改为流式播放
4. 添加相机切换按钮
5. **新增**: 添加 Rerun 可视化按钮和功能入口

## 待扩展功能

### 已完成
- [x] **Rerun 数据可视化** - 支持 .rrd 文件生成和远程服务器模式

### 待开发
1. [ ] 批量操作 (批量删除、批量导出)
2. [ ] 视频质量档位切换 (360p/480p/720p/1080p)
3. [ ] 视频下载 (完整打包下载)
4. [ ] 数据集统计面板 (可视化图表)
5. [ ] 高级搜索 (多条件组合)
6. [ ] 标签系统
7. [ ] 用户权限管理
8. [ ] 实时索引进度 WebSocket 推送
9. [ ] 视频标注功能
10. [ ] 数据集版本管理
