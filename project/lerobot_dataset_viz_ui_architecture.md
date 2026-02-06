# LeRobot 数据集可视化 UI 架构

## 目标
- 做一个独立 UI 用于 episode 级别可视化，前后端完全分离。
- 复用 `src/lerobot/scripts/lerobot_dataset_viz.py` 的读取逻辑与 `LeRobotDataset`。
- 原始数据保存在本地磁盘，元信息保存在数据库（`init.sql`）。
- API 以 `111.yaml` 为唯一契约，前端不感知存储细节。

## 范围与约束
- Item 即 episode。
- 自研可视化，不使用 Rerun Viewer。
- 本地磁盘是数据源；服务端只暴露下载/预览 URL。

## 组件划分
- 前端 SPA：数据集列表、episode 列表、播放器与曲线。
- API 服务：建议 FastAPI，实现 `111.yaml` 全部接口，内部调用 `LeRobotDataset`。
- 异步 Worker：解析与索引 episode，生成缩略图/预览资源。
- 存储：本地磁盘 + SQL（元信息）；可选 Nginx 提供静态大文件。

## 本地数据结构（LeRobotDataset）
数据根目录与 `datasets.file_path` 对应，常见路径如下：
- `meta/info.json`, `meta/episodes/*.parquet`, `meta/stats.json`
- `data/chunk-xxx/file-xxx.parquet`（动作/状态/时间戳）
- `videos/{camera_key}/chunk-xxx/file-xxx.mp4`（若启用视频）
- `images/`（可选帧图缓存）

解析与取数建议统一通过 `LeRobotDataset`，避免手写解析逻辑。

## 数据库模型
现有表：`datasets`, `robot_types`, `tasks`, `dataset_robot_tasks`。

建议新增表：
- `items`（episode 元信息）
  - 推荐字段：`id`, `dataset_id`, `episode_index`, `filename`, `content_type`, `size_bytes`,
    `frame_count`, `duration_s`, `file_path`, `created_at`, `deleted_at`
  - `filename` 可用 `episode-000123` 风格，`file_path` 指向 episode 在本地的根路径或资源索引文件。
- `item_assets`（派生资源）
  - `id`, `item_id`, `type`(thumbnail|preview_video|signals), `path`, `content_type`,
    `size_bytes`, `created_at`
- 可选：`item_signals`（若一定要 DB 级索引帧数据）；否则将曲线数据打包为
  parquet/npz，存为 `item_assets` 的 `signals` 资源。

推荐建表 SQL（MySQL/MariaDB 风格）：
```sql
CREATE TABLE IF NOT EXISTS items (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  dataset_id BIGINT NOT NULL,
  episode_index INT NOT NULL,
  filename VARCHAR(255) NOT NULL,
  content_type VARCHAR(100),
  size_bytes BIGINT,
  frame_count INT,
  duration_s DOUBLE,
  file_path VARCHAR(1000) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  deleted_at TIMESTAMP NULL,
  UNIQUE KEY uk_dataset_episode (dataset_id, episode_index),
  INDEX idx_dataset (dataset_id),
  INDEX idx_deleted_at (deleted_at),
  CONSTRAINT fk_items_dataset FOREIGN KEY (dataset_id)
    REFERENCES datasets(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='数据集 Episode 表';

CREATE TABLE IF NOT EXISTS item_assets (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  item_id BIGINT NOT NULL,
  type VARCHAR(50) NOT NULL,
  path VARCHAR(1000) NOT NULL,
  content_type VARCHAR(100),
  size_bytes BIGINT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_item (item_id),
  INDEX idx_item_type (item_id, type),
  CONSTRAINT fk_assets_item FOREIGN KEY (item_id)
    REFERENCES items(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Episode 派生资源';
```

## API 对应关系（基于 111.yaml）
- `/datasets` 与 `/datasets/{datasetId}`：`datasets` 的 CRUD。
- `/datasets/parse-lerobot`：校验本地路径并返回摘要（相机键、帧数、时长等）。
- `/datasets/create-lerobot`：创建数据集记录并触发索引任务。
- `/datasets/{datasetId}/items`：从 `items` 分页返回 episode 列表。
- `/items/{itemId}`：episode 详情。
- `/items/{itemId}/preview`：缩略图 URL + 元信息（fps、frame_count、camera_keys）。
- `/items/{itemId}/download`：返回视频/帧包/信号包的下载地址或直链。
- `/v1/records`：如需帧级曲线，可将 `values` 存为时序数据并按 `episode_id` 过滤。

## 可视化流程
1. 数据集列表：`/datasets`
2. Episode 列表：`/datasets/{datasetId}/items`
3. 预览：`/items/{itemId}/preview`（缩略图 + meta）
4. 播放：`/items/{itemId}/download`（视频或帧包 + signals）
5. 曲线与时间轴：对齐 `frame_index` 与 `timestamp`，展示动作/状态/奖励等。

## 后台任务
- 扫描 `meta/episodes` 与 `data/`，生成 `items` 记录。
- 计算 `frame_count`、`duration_s`、size 等统计。
- 生成缩略图、低清预览视频或 signals 包。
- 写入 `item_assets`，并更新 `items` 的派生字段。

## 安全与访问控制
- 所有 dataset/item 接口需鉴权（`111.yaml` 的 bearerAuth）。
- API 返回 URL，不暴露本地绝对路径。
- 校验 `file_path`，防止路径穿越与任意文件读取。
