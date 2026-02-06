-- ============================================
-- 机器人数据集管理系统 - 数据库初始化脚本
-- ============================================

-- 1. 创建数据库（可选，如需要请取消注释并修改数据库名）
-- CREATE DATABASE IF NOT EXISTS lerobot_dataset DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;
-- USE lerobot_dataset;

-- ============================================
-- 2. 创建表结构
-- ============================================

-- 机器人类型表
CREATE TABLE IF NOT EXISTS robot_types (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL UNIQUE COMMENT '机器人类型名称，如 agilex, 智元',
    description TEXT COMMENT '机器人类型描述',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='机器人类型表';

-- 任务类型表
CREATE TABLE IF NOT EXISTS tasks (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL UNIQUE COMMENT '任务名称，如 家居, 办公, 工业',
    description TEXT COMMENT '任务描述',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='任务类型表';

-- 数据集表
CREATE TABLE IF NOT EXISTS datasets (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL COMMENT '数据集名称',
    file_path VARCHAR(1000) NOT NULL COMMENT '数据集文件存储路径',
    description TEXT COMMENT '数据集详细描述',

    -- 元数据字段
    file_size BIGINT COMMENT '文件大小（字节）',
    sample_count INT COMMENT '样本数量',
    version VARCHAR(50) COMMENT '版本号',
    format VARCHAR(50) COMMENT '数据格式，如 hdf5, zip, parquet',

    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_name (name),
    INDEX idx_file_path (file_path(255)),
    INDEX idx_version (version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='数据集表';

-- 多对多关联表
CREATE TABLE IF NOT EXISTS dataset_robot_tasks (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    dataset_id BIGINT NOT NULL,
    robot_type_id BIGINT NOT NULL,
    task_id BIGINT NOT NULL,

    -- 关联时的额外信息
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 联合唯一索引
    UNIQUE KEY uk_dataset_robot_task (dataset_id, robot_type_id, task_id),

    -- 外键约束
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (robot_type_id) REFERENCES robot_types(id) ON DELETE CASCADE,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE,

    INDEX idx_dataset (dataset_id),
    INDEX idx_robot (robot_type_id),
    INDEX idx_task (task_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='数据集-机器人-任务关联表';

-- ============================================
-- 3. 初始化机器人类型数据
-- ============================================

INSERT INTO robot_types (name, description) VALUES
-- 国内机器人品牌
('agilex', 'Agilex Robotics - 主要生产移动机器人底盘，如 Scout, Hunter 系列'),
('智元', '智元机器人 - 远征A1系列人形机器人'),
('unitree', '宇树科技 - 四足机器人（Go系列）、人形机器人（H1）'),
('xiaomi', '小米 - CyberOne 人形机器人'),
('lexitech', '乐聚机器人 - Aelos、Talos 人形机器人'),
('deeprobotics', '云深处科技 - 四足机器人（绝影系列）'),
('fourier', '傅利叶智能 - GR-1 通用人形机器人'),

-- 国外知名机器人品牌
('fetch', 'Fetch Robotics - Fetch Freight 移动操作机器人'),
('kuka', 'KUKA - 工业机械臂，iiwa、LBR 系列'),
('abb', 'ABB - 工业机器人及协作机器人'),
('universal_robots', 'Universal Robots - UR系列协作机械臂'),
('franka', 'Franka Emika - Panda 协作机械臂，广泛用于研究'),
('boston_dynamics', 'Boston Dynamics - Spot 四足机器人、Atlas 人形机器人'),
('shadow', 'Shadow Robot Company - 灵巧手及机械臂系统'),
('willow_garage', 'Willow Garage - PR2 个人机器人'),
('pal_robotics', 'PAL Robotics - REEM、TIAGo 系列机器人'),
('softbank', 'SoftBank Robotics - Pepper、NAO 人形机器人'),
('dyson', 'Dyson - 360 Heurist 吸尘机器人'),
('figure', 'Figure AI - Figure 01/02 人形机器人'),
('apptronik', 'Apptronik - Aria 人形机器人'),
('agility', 'Agility Robotics - Digit 双足机器人'),
('sanctuary', 'Sanctuary AI - Phoenix 通用机器人'),
('tesla', 'Tesla - Optimus 人形机器人'),
('mujin', 'Mujin - 工业自动化机器人控制器'),
('rt', 'Rethink Robotics - Baxter、Sawyer 协作机器人'),

-- 开源/研究平台
('aloha', 'ALOHA - 低成本远程操作硬件平台'),
('bridge', 'BridgeData - 数据集采集机器人平台'),
('droid', 'DROID - 分布式机器人数据采集平台'),

-- 专用机器人类型
('manipulator', '通用机械臂'),
('mobile_base', '移动底盘'),
('gripper', '夹爪/末端执行器'),
('humanoid', '人形机器人'),
('quadruped', '四足机器人'),
('dual_arm', '双臂机器人'),
('scara', 'SCARA 机器人'),
('delta', 'Delta 并联机器人'),
('cobot', '协作机器人');

-- ============================================
-- 4. 初始化任务类型数据（示例）
-- ============================================

INSERT INTO tasks (name, description) VALUES
('家居', '家庭场景任务：叠衣服、整理物品、清洁等'),
('办公', '办公场景任务：文件整理、物品传递、桌面清理等'),
('工业', '工业场景任务：装配、搬运、分拣、质检等'),
('服务', '服务场景任务：接待、引导、配送等'),
('医疗', '医疗场景任务：手术辅助、药品配送、康复训练等'),
('农业', '农业场景任务：采摘、播种、喷洒等'),
('仓储', '仓储场景任务：拣货、上架、盘点等'),
('烹饪', '烹饪场景任务：备菜、烹饪、摆盘等'),
('探索', '探索场景任务：导航、地形识别、目标搜索等'),
('遥操作', '远程操作任务：通过遥操作收集演示数据');

-- ============================================
-- 查询示例
-- ============================================

-- 查看所有机器人类型
-- SELECT * FROM robot_types ORDER BY id;

-- 查看所有任务类型
-- SELECT * FROM tasks ORDER BY id;

-- 查询某个机器人类型对应的所有数据集
-- SELECT d.name, d.file_path, rt.name as robot_type, t.name as task
-- FROM datasets d
-- JOIN dataset_robot_tasks drt ON d.id = drt.dataset_id
-- JOIN robot_types rt ON drt.robot_type_id = rt.id
-- JOIN tasks t ON drt.task_id = t.id
-- WHERE rt.name = 'agilex';

-- 添加新数据集并关联机器人和任务的示例
-- START TRANSACTION;
-- INSERT INTO datasets (name, file_path, description, format) VALUES
-- ('家居抓取演示数据集', '/data/datasets/home_grasp_v1', '用于家居场景抓取任务的演示数据', 'hdf5');
-- SET @dataset_id = LAST_INSERT_ID();
-- INSERT INTO dataset_robot_tasks (dataset_id, robot_type_id, task_id)
-- SELECT @dataset_id, id, (SELECT id FROM tasks WHERE name = '家居')
-- FROM robot_types WHERE name = 'agilex';
-- COMMIT;
