#!/usr/bin/env python
"""Initialize SQLite database for LeRobot Dataset Management System."""
import sqlite3
import os

# Database path
db_path = r'D:\lerobot_package\lerobot_dataset.db'
os.makedirs(os.path.dirname(db_path), exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Enable foreign keys
cursor.execute("PRAGMA foreign_keys = ON")

# Create tables
sql_statements = [
    # Robot types table
    '''CREATE TABLE IF NOT EXISTS robot_types (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''',

    # Tasks table
    '''CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''',

    # Datasets table
    '''CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        description TEXT,
        file_size INTEGER,
        sample_count INTEGER,
        version TEXT,
        format TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''',

    # Dataset-robot-tasks association table
    '''CREATE TABLE IF NOT EXISTS dataset_robot_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL,
        robot_type_id INTEGER NOT NULL,
        task_id INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (dataset_id, robot_type_id, task_id),
        FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
        FOREIGN KEY (robot_type_id) REFERENCES robot_types(id) ON DELETE CASCADE,
        FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
    )''',

    # Items (episode) table
    '''CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL,
        episode_index INTEGER NOT NULL,
        filename TEXT NOT NULL,
        content_type TEXT,
        size_bytes INTEGER,
        frame_count INTEGER,
        duration_s REAL,
        file_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        deleted_at TIMESTAMP NULL,
        UNIQUE (dataset_id, episode_index),
        FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
    )''',

    # Item assets table
    '''CREATE TABLE IF NOT EXISTS item_assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_id INTEGER NOT NULL,
        type TEXT NOT NULL,
        path TEXT NOT NULL,
        content_type TEXT,
        size_bytes INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE CASCADE
    )'''
]

for sql in sql_statements:
    cursor.execute(sql)

# Insert robot types
robot_types = [
    ('agilex', 'Agilex Robotics - Mobile robot chassis'),
    ('zhiyuan', 'Zhiyuan Robot - Yuanzheng A1 humanoid'),
    ('unitree', 'Unitree - Quadruped and humanoid robots'),
    ('xiaomi', 'Xiaomi - CyberOne humanoid'),
    ('lexitech', 'Leju Robot - Aelos, Talos humanoid'),
    ('deeprobotics', 'DeepRobotics - Jueying quadruped'),
    ('fourier', 'Fourier Intelligence - GR-1 humanoid'),
    ('fetch', 'Fetch Robotics - Fetch Freight'),
    ('kuka', 'KUKA - Industrial robotic arms'),
    ('abb', 'ABB - Industrial robots'),
    ('universal_robots', 'Universal Robots - UR cobots'),
    ('franka', 'Franka Emika - Panda cobot'),
    ('boston_dynamics', 'Boston Dynamics - Spot, Atlas'),
    ('shadow', 'Shadow Robot - Dexterous hands'),
    ('willow_garage', 'Willow Garage - PR2'),
    ('pal_robotics', 'PAL Robotics - REEM, TIAGo'),
    ('softbank', 'SoftBank - Pepper, NAO'),
    ('dyson', 'Dyson - 360 Heurist vacuum'),
    ('figure', 'Figure AI - Figure 01/02'),
    ('apptronik', 'Apptronik - Aria humanoid'),
    ('agility', 'Agility Robotics - Digit biped'),
    ('sanctuary', 'Sanctuary AI - Phoenix'),
    ('tesla', 'Tesla - Optimus humanoid'),
    ('mujin', 'Mujin - Industrial controllers'),
    ('rt', 'Rethink Robotics - Baxter, Sawyer'),
    ('aloha', 'ALOHA - Low-cost teleoperation'),
    ('bridge', 'BridgeData - Data collection'),
    ('droid', 'DROID - Distributed robot data'),
    ('manipulator', 'General manipulator'),
    ('mobile_base', 'Mobile base'),
    ('gripper', 'Gripper/End-effector'),
    ('humanoid', 'Humanoid robot'),
    ('quadruped', 'Quadruped robot'),
    ('dual_arm', 'Dual-arm robot'),
    ('scara', 'SCARA robot'),
    ('delta', 'Delta parallel robot'),
    ('cobot', 'Collaborative robot')
]

cursor.executemany('INSERT OR IGNORE INTO robot_types (name, description) VALUES (?, ?)', robot_types)

# Insert tasks
tasks = [
    ('home', 'Home tasks: folding, organizing, cleaning'),
    ('office', 'Office tasks: filing, passing items, desk cleaning'),
    ('industrial', 'Industrial tasks: assembly, handling, sorting, QC'),
    ('service', 'Service tasks: reception, guidance, delivery'),
    ('medical', 'Medical tasks: surgery assist, medicine delivery'),
    ('agriculture', 'Agriculture tasks: harvesting, seeding, spraying'),
    ('warehouse', 'Warehouse tasks: picking, stocking, inventory'),
    ('cooking', 'Cooking tasks: prep, cooking, plating'),
    ('exploration', 'Exploration tasks: navigation, search'),
    ('teleoperation', 'Teleoperation: data collection via remote control')
]

cursor.executemany('INSERT OR IGNORE INTO tasks (name, description) VALUES (?, ?)', tasks)

conn.commit()

# Print summary
print(f'Database created: {db_path}')
print(f'robot_types: {cursor.execute("SELECT COUNT(*) FROM robot_types").fetchone()[0]} rows')
print(f'tasks: {cursor.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]} rows')

conn.close()
print('Database initialization complete!')
