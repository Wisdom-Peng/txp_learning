import numpy as np
import torch
import triton

def print_environment_info():
    try:
        # 打印 PyTorch 和 CUDA 信息
        print("=" * 40)
        print("[PyTorch Environment]")
        print(f"PyTorch Version: {torch.__version__}")  # 修正：版本属性为小写 __version__
        print(f"CUDA Available: {torch.cuda.is_available()}")  # 修正：拼写错误 is_available()

        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA Not Available")

        # 打印 NumPy 信息
        print("\n[NumPy Environment]")
        print(f"NumPy Version: {np.__version__}")
        # print("=" * 40)

        # 打印 triton 环境
        print("\n[triton Environment]")
        print(f"triton Version: {triton.__version__}")
        print("=" * 40)

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    try:
        # 检查库是否安装
        print_environment_info()
    except ImportError as e:
        print(f"Missing Library: {e.name}")
        print("Please install with:")
        print(f"pip install {e.name.replace('_', '-')}")