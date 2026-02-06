#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LeRobot 数据集可视化独立脚本

这个脚本提供了与 lerobot-dataset-viz 相同的功能，但可以独立运行。

功能说明：
    可视化 LeRobotDataset 类型数据集中任意情节的所有帧数据。
    支持图像、动作、状态、奖励等多种数据类型的可视化。

依赖：
    pip install rerun torch numpy tqdm
    pip install lerobot  # 需要安装 lerobot 包

使用示例：

1. 本地可视化数据集：
   python standalone_dataset_viz.py \
       --repo-id lerobot/pusht \
       --episode-index 0

2. 保存为 .rrd 文件（用于本地查看）：
   python standalone_dataset_viz.py \
       --repo-id lerobot/pusht \
       --episode-index 0 \
       --save 1 \
       --output-dir ./output

   然后在本地查看：
   rerun ./output/lerobot_pusht_episode_0.rrd

3. 远程机器上通过流式传输查看：
   （需要转发 websocket 端口到远程机器）
   ssh -L 9087:localhost:9087 username@remote-host

   在远程机器上运行：
   python standalone_dataset_viz.py \
       --repo-id lerobot/pusht \
       --episode-index 0 \
       --mode distant \
       --ws-port 9087

   在本地机器上运行：
   rerun ws://localhost:9087

4. 使用本地数据集（推荐）：
   python standalone_dataset_viz.py \
       --repo-id so101_v3_dataset1_clean \
       --episode-index 0 \
       --root D: \
       --local

python standalone_dataset_viz.py --repo-id D:\so101_v3_dataset1_clean --episode-index 0 --root  "D:\so101_v3_dataset1_clean" --local

   当使用 --local 参数时，脚本不会尝试连接 HuggingFace，
   直接从本地加载数据集。适用于自定义/本地采集的数据集。

参数说明：
    --repo-id: 数据集名称或 HuggingFace 仓库 ID（例如：lerobot/pusht）
    --episode-index: 要可视化的情节索引
    --root: 本地数据集根目录（使用 --local 时必需）
    --output-dir: 输出 .rrd 文件的目录
    --batch-size: DataLoader 的批次大小（默认：32）
    --num-workers: 数据加载的工作进程数（默认：4）
    --mode: 查看模式，'local' 或 'distant'（默认：local）
    --web-port: rerun 的 Web 端口（默认：9090）
    --ws-port: rerun 的 WebSocket 端口（默认：9087）
    --save: 是否保存为 .rrd 文件（0 或 1，默认：0）
    --tolerance-s: 时间戳容差（秒，默认：1e-4）
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """检查所需的依赖包是否已安装"""
    missing_deps = []

    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")

    try:
        import torch
    except ImportError:
        missing_deps.append("torch")

    try:
        import rerun
    except ImportError:
        missing_deps.append("rerun")

    try:
        import tqdm
    except ImportError:
        missing_deps.append("tqdm")

    try:
        import lerobot
    except ImportError:
        missing_deps.append("lerobot")

    if missing_deps:
        logger.error("缺少以下依赖包:")
        for dep in missing_deps:
            logger.error(f"  - {dep}")
        logger.error("\n请运行以下命令安装依赖:")
        if "lerobot" in missing_deps:
            logger.error("  pip install lerobot")
        logger.error("  pip install rerun torch numpy tqdm")
        sys.exit(1)

    logger.info("所有依赖检查通过 ✓")


def to_hwc_uint8_numpy(chw_float32_torch):
    """
    将 CHW 格式的 float32 torch tensor 转换为 HWC 格式的 uint8 numpy array

    参数:
        chw_float32_torch: 形状为 (C, H, W) 的 torch.Tensor，值范围 [0, 1]

    返回:
        hwc_uint8_numpy: 形状为 (H, W, C) 的 numpy array，值范围 [0, 255]
    """
    import torch
    import numpy as np

    assert chw_float32_torch.dtype == torch.float32, f"期望 float32，得到 {chw_float32_torch.dtype}"
    assert chw_float32_torch.ndim == 3, f"期望 3 维，得到 {chw_float32_torch.ndim}"
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"期望通道在前的图像格式，但得到 {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def visualize_dataset(
    dataset,
    repo_id: str,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
    jpeg_quality: int = 85,
    scale_factor: float | None = None,
):
    """
    可视化数据集

    参数:
        dataset: LeRobotDataset 实例
        repo_id: 数据集名称
        episode_index: 要可视化的情节索引
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        mode: 'local' 或 'distant'
        web_port: Web 端口（distant 模式）
        ws_port: WebSocket 端口（distant 模式）
        save: 是否保存为 .rrd 文件
        output_dir: 输出目录
        jpeg_quality: JPEG 压缩质量 (1-100, 默认85，越低文件越小)
        scale_factor: 图像缩放因子 (默认1.0=原尺寸, 0.5=半尺寸)

    返回:
        如果 save=True，返回 .rrd 文件路径
    """
    import torch
    import tqdm
    import rerun as rr

    if save:
        assert output_dir is not None, (
            "请使用 --output-dir 设置输出目录来保存 .rrd 文件"
        )

    # repo_id 现在作为参数传入，不再从 dataset 获取

    logger.info("正在加载数据加载器 (DataLoader)...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    logger.info(f"启动 Rerun 可视化，模式: {mode}")

    if mode not in ["local", "distant"]:
        raise ValueError(f"无效的模式: {mode}。必须是 'local' 或 'distant'")

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # 手动调用 Python 垃圾回收器，避免在 num_workers > 0 时挂起
    # TODO: 当 rerun 0.16 版本发布后移除此 gc.collect
    gc.collect()

    if mode == "distant":
        logger.info(f"启动 Web 服务器，端口: {web_port}")
        logger.info(f"WebSocket 端口: {ws_port}")
        rr.serve_web_viewer(open_browser=False, web_port=web_port)

    logger.info("开始记录数据到 Rerun...")

    # 定义常量
    ACTION = "action"
    DONE = "done"
    OBS_STATE = "observation.state"
    REWARD = "reward"

    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="可视化进度"):
        # 遍历批次中的每个样本
        for i in range(len(batch["index"])):
            frame_index = batch["frame_index"][i].item()
            timestamp = batch["timestamp"][i].item()

            rr.set_time("frame_index", sequence=frame_index)
            rr.set_time("timestamp", timestamp=timestamp)

            # 显示每个相机图像（带压缩）
            if hasattr(dataset.meta, 'camera_keys'):
                for key in dataset.meta.camera_keys:
                    if key in batch:
                        img_array = to_hwc_uint8_numpy(batch[key][i])
                        # 如果指定了缩放比例，缩小图像尺寸
                        if scale_factor is not None and scale_factor < 1.0:
                            import cv2
                            h, w = img_array.shape[:2]
                            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                            img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        # 使用 JPEG 压缩
                        rr.log(key, rr.Image(img_array).compress(jpeg_quality=jpeg_quality))
            else:
                # 尝试自动检测图像键
                for key in batch.keys():
                    if 'image' in key.lower() or 'camera' in key.lower():
                        if isinstance(batch[key], torch.Tensor) and batch[key].dim() == 4:
                            img_array = to_hwc_uint8_numpy(batch[key][i])
                            # 如果指定了缩放比例，缩小图像尺寸
                            if scale_factor is not None and scale_factor < 1.0:
                                import cv2
                                h, w = img_array.shape[:2]
                                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                                img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            # 使用 JPEG 压缩
                            rr.log(key, rr.Image(img_array).compress(jpeg_quality=jpeg_quality))

            # 显示动作空间的每个维度（例如：执行器命令）
            if ACTION in batch:
                for dim_idx, val in enumerate(batch[ACTION][i]):
                    rr.log(f"{ACTION}/{dim_idx}", rr.Scalars(val.item()))

            # 显示观测状态空间的每个维度（例如：关节空间中的智能体位置）
            if OBS_STATE in batch:
                for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))

            # 显示 done 标志
            if DONE in batch:
                rr.log(DONE, rr.Scalars(batch[DONE][i].item()))

            # 显示奖励
            if REWARD in batch:
                rr.log(REWARD, rr.Scalars(batch[REWARD][i].item()))

            # 显示成功标志
            if "next.success" in batch:
                rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))

    logger.info("数据记录完成！")

    if mode == "local" and save:
        # 在本地保存 .rrd 文件
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        logger.info(f"保存 .rrd 文件到: {rrd_path}")
        rr.save(rrd_path)
        logger.info(f"✓ 文件已保存！使用以下命令查看:")
        logger.info(f"  rerun {rrd_path}")
        return rrd_path

    elif mode == "distant":
        logger.info("远程服务器正在运行...")
        logger.info("按 Ctrl+C 停止服务器")
        # 防止进程退出，因为它正在提供 websocket 连接
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到 Ctrl-C。退出。")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="LeRobot 数据集可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  从 HuggingFace 加载并可视化:
    python %(prog)s --repo-id lerobot/pusht --episode-index 0

  可视化本地数据集（推荐）:
    python %(prog)s --repo-id my_dataset --episode-index 0 --root D:/data/my_dataset --local

    注意: 使用 --local 参数可以避免连接 HuggingFace，适用于自定义数据集

  保存为 .rrd 文件:
    python %(prog)s --repo-id my_dataset --episode-index 0 --root ./data --local --save 1 --output-dir ./output

  远程模式（数据在服务器）:
    python %(prog)s --repo-id my_dataset --episode-index 0 --root ./data --local --mode distant --ws-port 9087
        """
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace 仓库 ID，包含 LeRobotDataset 数据集（例如：lerobot/pusht）"
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="要可视化的情节索引"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="本地存储的数据集根目录（例如：--root data）。默认情况下，将从 hugging face 缓存文件夹加载数据集，或从 hub 下载（如果可用）。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="当设置 --save 1 时，写入 .rrd 文件的目录路径"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="DataLoader 加载的批次大小（默认：32）"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader 加载数据的进程数（默认：4）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local", "distant"],
        help="查看模式：'local' 或 'distant'。'local' 要求数据在本地机器上。'distant' 在存储数据的远程机器上创建服务器。"
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="当设置 --mode distant 时，rerun.io 的 Web 端口（默认：9090）"
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="当设置 --mode distant 时，rerun.io 的 WebSocket 端口（默认：9087）"
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        choices=[0, 1],
        help="在 --output-dir 提供的目录中保存 .rrd 文件。这也会停用查看器的启动。在本地机器上运行 `rerun path/to/file.rrd` 来查看数据。"
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="用于确保数据时间戳遵守数据集 fps 值的容差（秒）。这是传递给 LeRobotDataset 构造函数的参数。（默认：1e-4）"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="使用纯本地模式，不尝试连接 HuggingFace。适用于本地采集或自定义的数据集。需要提供 --root 参数。"
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG 压缩质量 (1-100, 默认85)。越低文件越小，但画质越差。推荐: 70-90"
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help="图像缩放因子 (默认1.0=原尺寸)。例如: 0.5=半尺寸, 0.25=四分之一尺寸。可显著减小文件大小。"
    )

    args = parser.parse_args()

    # 验证参数
    if args.local and args.root is None:
        logger.error("使用 --local 参数时，必须提供 --root 参数指定数据集路径")
        sys.exit(1)

    # 检查依赖
    check_dependencies()

    # 使用本地模式，避免连接 HuggingFace
    # 注意：必须在导入 LeRobotDataset 之前设置环境变量
    if args.local:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        logger.info("🔒 使用纯本地模式，不会连接 HuggingFace")

    # 导入 lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # 加载数据集
    logger.info(f"正在加载数据集: {args.repo_id}")
    logger.info(f"情节索引: {args.episode_index}")

    if args.root:
        logger.info(f"数据集根目录: {args.root}")

    try:
        # 构建数据集加载参数
        dataset_kwargs = {
            "repo_id": args.repo_id,
            "episodes": [args.episode_index],
            "tolerance_s": args.tolerance_s,
        }

        if args.root:
            dataset_kwargs["root"] = args.root

        dataset = LeRobotDataset(**dataset_kwargs)
        logger.info(f"✓ 数据集加载成功！")
        logger.info(f"  情节数量: {dataset.num_episodes}")
        logger.info(f"  总帧数: {dataset.num_frames}")

        # 显示数据集信息
        if hasattr(dataset.meta, 'info'):
            logger.info(f"  FPS: {dataset.meta.fps if hasattr(dataset.meta, 'fps') else 'N/A'}")

    except FileNotFoundError as e:
        logger.error(f"❌ 数据集文件未找到: {e}")
        if args.root:
            logger.error(f"请检查 --root 路径是否正确: {args.root}")
            logger.error("预期路径格式: --root D:/datasets/your_dataset")
            logger.error("或: --root /path/to/datasets/your_dataset")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 加载数据集失败: {e}")

        # 提供更详细的错误信息
        if "huggingface.co" in str(e) or "Connection" in str(e):
            logger.error("\n💡 这看起来是网络连接问题。")
            logger.error("如果您使用的是本地数据集，请添加 --local 参数:")
            logger.error(f"  python {sys.argv[0]} --repo-id {args.repo_id} --episode-index {args.episode_index} --root YOUR_DATA_PATH --local")
        else:
            logger.error("请检查以下内容:")
            logger.error("  1. --repo-id 是否正确")
            logger.error("  2. --episode-index 是否在有效范围内")
            logger.error("  3. --root 路径是否正确")
            logger.error("  4. 数据集是否完整（包含 meta/info.json 等文件）")
        sys.exit(1)

    # 准备可视化参数
    viz_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "mode": args.mode,
        "web_port": args.web_port,
        "ws_port": args.ws_port,
        "save": bool(args.save),
        "output_dir": args.output_dir,
        "jpeg_quality": args.jpeg_quality,
        "scale_factor": args.scale_factor,
    }

    # 开始可视化
    try:
        result = visualize_dataset(dataset, args.repo_id, args.episode_index, **viz_kwargs)
        if result:
            logger.info(f"✓ 可视化完成！文件已保存到: {result}")
        elif args.mode == "local":
            logger.info("✓ 可视化完成！查看器应该已自动打开。")
    except Exception as e:
        logger.error(f"可视化过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
