-- Items (episode) table
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

-- Derived assets (thumbnails, etc.)
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
