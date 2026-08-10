# convert_hdf5_to_lerobot.py  
import h5py  
import numpy as np  
from pathlib import Path  
from tqdm import tqdm  # 添加进度条库  
from lerobot.datasets.lerobot_dataset import LeRobotDataset  
  
  
def convert_kinova_hdf5_to_lerobot(  
    hdf5_path: str,  
    output_dir: str,  
    fps: int = 30,  
):  
    """  
    将 Kinova Gen3 HDF5 数据转换为 LeRobot Dataset 格式(纯本地,不与 Hub 交互)  
      
    Args:  
        hdf5_path: HDF5 文件路径  
        output_dir: 本地输出目录  
        fps: 帧率  
      
    Returns:  
        LeRobotDataset: 转换后的数据集  
    """  
    local_repo_id = "local/kinova_pickplace"  
      
    # 1. 定义数据集特征  
    features = {  
        "observation.images.agentview": {  
            "dtype": "video",  
            "shape": (3, 84, 84),  
            "names": ["channels", "height", "width"],  
        },  
        "observation.images.wrist": {  
            "dtype": "video",  
            "shape": (3, 84, 84),  
            "names": ["channels", "height", "width"],  
        },  
        "observation.state": {  
            "dtype": "float32",  
            "shape": (21,),  
            "names": None,  
        },  
        "action": {  
            "dtype": "float32",  
            "shape": (7,),  
            "names": None,  
        },  
        "reward": {  
            "dtype": "float32",  
            "shape": (1,),  
            "names": None,  
        },  
        "done": {  
            "dtype": "bool",  
            "shape": (1,),  
            "names": None,  
        },  
    }  
  
    # 2. 创建空的 LeRobot 数据集  
    dataset = LeRobotDataset.create(  
        repo_id=local_repo_id,  
        fps=fps,  
        root=output_dir,  
        robot_type="kinova_gen3",  
        features=features,  
        use_videos=True,  
    )  
  
    # 3. 启动图像写入器  
    dataset.start_image_writer(num_processes=0, num_threads=3)  
  
    # 4. 读取 HDF5 并逐帧添加  
    with h5py.File(hdf5_path, 'r') as f:  
        demos = sorted(list(f['data'].keys()),  
                      key=lambda x: int(x.split('_')[1]))  
  
        print(f"开始转换 {len(demos)} 个演示...")  
          
        # 添加演示级别的进度条  
        for demo_idx, demo in enumerate(tqdm(demos, desc="转换演示", unit="demo")):  
            demo_grp = f[f'data/{demo}']  
            obs_grp = demo_grp['obs']  
  
            num_frames = demo_grp['actions'].shape[0]  
              
            # 添加帧级别的进度条  
            for i in tqdm(range(num_frames), desc=f"  {demo}", leave=False, unit="frame"):  
                # 读取图像并转换为 CHW 格式  
                agentview_img = obs_grp['agentview_image'][i]  
                wrist_img = obs_grp['robot0_eye_in_hand_image'][i]  
                  
                # 转置为 (C, H, W)  
                agentview_img = np.transpose(agentview_img, (2, 0, 1))  
                wrist_img = np.transpose(wrist_img, (2, 0, 1))  
  
                frame_dict = {  
                    "observation.images.agentview": agentview_img,  
                    "observation.images.wrist": wrist_img,  
                    "observation.state": np.concatenate([  
                        obs_grp['robot0_joint_pos'][i],  
                        obs_grp['robot0_joint_vel'][i],  
                        obs_grp['robot0_eef_pos'][i],  
                        obs_grp['robot0_eef_quat'][i],  
                    ]).astype(np.float32),  
                    "action": demo_grp['actions'][i].astype(np.float32),  
                    "reward": np.array([demo_grp['rewards'][i]], dtype=np.float32),  
                    "done": np.array([demo_grp['dones'][i]], dtype=bool),  
                    "task": "pick_place_milk",  
                }  
  
                dataset.add_frame(frame_dict)  
  
            dataset.save_episode()  
  
    # 5. 停止图像写入并完成数据集  
    print("\n正在完成数据集...")  
    dataset.stop_image_writer()  
    dataset.finalize()  
      
    print(f"✓ 数据集转换完成!")  
    print(f"  - 总演示数: {dataset.meta.total_episodes}")  
    print(f"  - 总帧数: {dataset.meta.total_frames}")  
    print(f"  - 存储路径: {dataset.root}")  
  
    return dataset  
  
  
def main():  
    """主函数:转换 data/merge_300/image.hdf5 文件"""  
    hdf5_path = "data/merge_300/image.hdf5"  
    output_dir = "./kinova_data"  
      
    print(f"开始转换 HDF5 数据集...")  
    print(f"输入文件: {hdf5_path}")  
    print(f"输出目录: {output_dir}\n")  
      
    dataset = convert_kinova_hdf5_to_lerobot(  
        hdf5_path=hdf5_path,  
        output_dir=output_dir,  
        fps=30,  
    )  
      
    print("\n数据集已保存到本地,无需上传到 Hub")  
      
    return dataset  
  
  
if __name__ == "__main__":  
    main()
