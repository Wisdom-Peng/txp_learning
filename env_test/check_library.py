import subprocess
import sys
import platform
import os
import shutil
from typing import Dict, Optional

def get_nvidia_driver_version() -> Optional[str]:
    """获取 NVIDIA 驱动版本"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split('\n')[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def get_gpu_info() -> Dict:
    """获取 GPU 设备信息"""
    info = {"devices": []}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.strip().split('\n'):
            name, mem = line.split(',')
            info["devices"].append({
                "name": name.strip(),
                "total_memory": mem.strip()
            })
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info

def check_library(lib_name: str) -> Dict:
    """检测指定库的安装状态及版本"""
    lib_checks = {
        # 深度学习框架
        "PyTorch": ("torch", lambda m: (m.__version__, m.version.cuda, m.backends.cudnn.version())),
        "TensorFlow": ("tensorflow", lambda m: (m.__version__, m.sysconfig.get_build_info()['cuda_version'])),
        "JAX": ("jax", lambda m: (m.__version__,)),

        # GPU 加速库
        "CUDA Toolkit": (
            "cuda",
            lambda m: (
                subprocess.getoutput("nvcc --version").split()[-2]
                if shutil.which("nvcc") else "Not Found"
            ),
        ),
        "cuDNN": (
            "cudnn",
            lambda m: (
                # 检查 cudnn 头文件或库路径
                open("/usr/local/cuda/include/cudnn_version.h").read().split()[-1].strip('"')
                if os.path.exists("/usr/local/cuda/include/cudnn_version.h") else "Not Found"
            ),
        ),
        # 其他工具库
        "NVIDIA APEX": ("apex", lambda m: (m.__version__,) if hasattr(m, "__version__") else ("unknown",)),
        "Triton": ("triton", lambda m: (m.__version__,)),
        "CuPy": ("cupy", lambda m: (m.__version__,)),
        "Numba": ("numba", lambda m: (m.__version__,)),
    }

    if lib_name not in lib_checks:
        return {"installed": False}

    module_name, version_getter = lib_checks[lib_name]
    try:
        __import__(module_name)
        module = sys.modules[module_name]
        versions = version_getter(module)
        
        # 处理特殊库
        details = {}
        if lib_name == "PyTorch":
            details = {
                "CUDA available": module.cuda.is_available(),
                "CUDA version": versions[1],
                "cuDNN version": versions[2]
            }
        elif lib_name == "cuDNN":
            details = {"version": versions[0]}

        return {
            "installed": True,
            "version": versions[0],
            **details
        }
    except ImportError:
        return {"installed": False}

def main():
    print("=" * 50)
    print("GPU Environment Diagnostic Report")
    print(f"Python: {platform.python_version()}")
    print(f"System: {platform.system()} {platform.release()}")
    print("=" * 50 + "\n")

    # 检测 NVIDIA 驱动
    driver_version = get_nvidia_driver_version()
    print(f"[NVIDIA Driver]")
    print(f"  - Version: {driver_version or 'Not Found'}\n")

    # 检测 GPU 设备
    gpu_info = get_gpu_info()
    print("[GPU Devices]")
    if gpu_info["devices"]:
        for idx, device in enumerate(gpu_info["devices"], 1):
            print(f"  GPU {idx}:")
            print(f"    - Name: {device['name']}")
            print(f"    - Memory: {device['total_memory']}")
    else:
        print("  - No NVIDIA GPU detected")
    print()

    # 检测关键库
    libraries = [
        "PyTorch", "TensorFlow", "JAX",
        "CUDA Toolkit", "cuDNN",
        "NVIDIA APEX", "Triton", "CuPy", "Numba"
    ]

    for lib in libraries:
        result = check_library(lib)
        print(f"[{lib}]")
        if result["installed"]:
            print(f"  - Installed: Yes")
            print(f"  - Version: {result.get('version', 'unknown')}")
            if lib == "PyTorch":
                print(f"  - CUDA Available: {result['CUDA available']}")
                print(f"  - CUDA Version: {result.get('CUDA version', 'N/A')}")
                print(f"  - cuDNN Version: {result.get('cuDNN version', 'N/A')}")
        else:
            print("  - Installed: No")
        print()

if __name__ == "__main__":
    main()
