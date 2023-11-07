import pathlib
from setuptools import setup, find_packages
import torch.utils.cpp_extension as torch_cpp_ext

root = pathlib.Path(__name__).parent
print(root)


def glob(pattern):
  return [str(p) for p in root.glob(pattern)]

def remove_unwanted_pytorch_nvcc_flags():
  REMOVE_NVCC_FLAGS = [
      '-D__CUDA_NO_HALF_OPERATORS__',
      '-D__CUDA_NO_HALF_CONVERSIONS__',
      '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
      '-D__CUDA_NO_HALF2_OPERATORS__',
  ]
  for flag in REMOVE_NVCC_FLAGS:
    try:
      torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
      pass


remove_unwanted_pytorch_nvcc_flags()
ext_modules = []
ext_modules.append(
    torch_cpp_ext.CUDAExtension(
        "slora._kernels",
        ["slora/csrc/lora_ops.cc"] +
        glob("slora/csrc/bgmv/*.cu"),
        extra_compile_args=['-std=c++17'],
    ))

setup(
    name="slora",
    version="1.0.0",
    packages=find_packages(
        exclude=("build", "include", "csrc", "test", "dist", "docs", "benchmarks", "slora.egg-info")
    ),
    author="model toolchain",
    author_email="",
    description="slora for inference LLM",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
    python_requires=">=3.9",
    install_requires=[
        "aiohttp",
        "einops",
        "fastapi",
        "ninja",
        "packaging",
        "pyzmq",
        "rpyc",
        "safetensors",
        # "torch",
        "transformers",
        "triton==2.1.0",
        "uvloop",
        "uvicorn",
        "psutil",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
)
