import h5py
import json
import os
import numpy as np


file_path = '/Users/skywalker/Downloads/two_arm_box_cleanup.hdf5'

# 以只读模式 ('r') 打开文件
with h5py.File(file_path, 'r') as f:
    
    # 1. 查看根目录下的第一层内容
    print("根目录内容:", list(f.keys()))
    
    # 2. 递归遍历并打印整个文件的所有结构路径
    print("\n--- 完整内部结构 ---")
    def print_structure(name, obj):
        # 打印名称以及它是 Group 还是 Dataset
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name} (Shape: {obj.shape}, Type: {obj.dtype})")
            
    f.visititems(print_structure)


with h5py.File(file_path, 'r') as f:
    dataset = f['data/demo_0']
    # 遍历并打印该数据集的所有属性
    print("数据集属性:")
    for key, value in dataset.attrs.items():
        print(f"  {key}: {value}")


with h5py.File(file_path, 'r') as f:
    # 假设数据存放在 'data/demo_0' 下，根据实际结构调整
    demo = f['data/demo_0']
    # 提取第一帧的 24维 action 向量
    flat_action = demo['actions'][0]
    print(f"Flattened Action (24D) 第一帧:\n{flat_action}\n")
    print("--- 比对 action_dict 中的各部分 ---")
    action_dict = demo['action_dict']
    # 遍历字典，打印每个组件的名字、维度和第一帧的值
    for key in action_dict.keys():
        val = action_dict[key][0]
        dim = len(val) if isinstance(val, (np.ndarray, list)) else 1
        print(f"Key: {key:<20} | Dim: {dim:<2} | Value: {val}")






OUTPUT_DIR = "/Users/skywalker/Downloads/"  # 导出的配置文件存放目录
def inspect_and_extract_attrs(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        print(f"🔍 正在检索 HDF5 文件: {hdf5_path}\n")
        # 1. 检查根节点的属性 (f.attrs)
        print("======== 1. 根节点元信息 (Root Attributes) ========")
        for key in f.attrs.keys():
            val = f.attrs[key]
            print(f"📌 Key 发现: '{key}'")
            # 尝试解析 JSON 格式的字符串配置（如 env_args, env_meta）
            if isinstance(val, str):
                try:
                    parsed_json = json.loads(val)
                    save_path = os.path.join(OUTPUT_DIR, f"{key}.json")
                    with open(save_path, 'w', encoding='utf-8') as jf:
                        json.dump(parsed_json, jf, indent=4, ensure_ascii=False)
                    print(f"   └─ 成功解析为 JSON 并保存至: {save_path}")
                    # 特别关注：提取 controller_configs
                    if "controller_configs" in parsed_json:
                        print("   └─ ✨ 找到 controller_configs 控制器配置！")
                        ctrl_path = os.path.join(OUTPUT_DIR, "composite_controller_config.json")
                        with open(ctrl_path, 'w', encoding='utf-8') as cf:
                            json.dump(parsed_json["controller_configs"], cf, indent=4)
                        print(f"   └─ ✨ 已单独导出 Composite Controller 配置至: {ctrl_path}")
                except json.JSONDecodeError:
                    # 如果是 XML 模型字符串 (如 model_file)
                    if "<mujoco" in val or "<worldbody" in val:
                        xml_path = os.path.join(OUTPUT_DIR, f"{key}.xml")
                        with open(xml_path, 'w', encoding='utf-8') as xf:
                            xf.write(val)
                        print(f"   └─ 成功识别为 MuJoCo XML 模型并保存至: {xml_path}")
                    else:
                        print(f"   └─ 字符串类型值 (前100字符): {val[:100]}...")
        # 2. 检查 'data' 节点或首个 Episode 节点的属性
        print("\n======== 2. 节点/Episode 层级元信息 ========")
        root_node = f['data'] if 'data' in f else f
        # 检查 data.attrs
        if hasattr(root_node, 'attrs') and len(root_node.attrs) > 0:
            for key in root_node.attrs.keys():
                print(f"📌 data.attrs Key 发现: '{key}'")
                val = root_node.attrs[key]
                if isinstance(val, str) and (val.startswith('{') or val.startswith('[')):
                    try:
                        parsed = json.loads(val)
                        save_path = os.path.join(OUTPUT_DIR, f"data_{key}.json")
                        with open(save_path, 'w') as jf:
                            json.dump(parsed, jf, indent=4)
                        print(f"   └─ 导出至: {save_path}")
                    except:
                        pass
        # 检查第一个 demo 是否含有独立的 model_file 或 config
        first_demo_key = list(root_node.keys())[0]
        first_demo = root_node[first_demo_key]
        if hasattr(first_demo, 'attrs') and 'model_file' in first_demo.attrs:
            xml_str = first_demo.attrs['model_file']
            xml_path = os.path.join(OUTPUT_DIR, "demo_model.xml")
            with open(xml_path, 'w') as xf:
                xf.write(xml_str)
            print(f"   └─ 从 {first_demo_key} 提取出 MuJoCo XML 保存至: {xml_path}")

inspect_and_extract_attrs(file_path)