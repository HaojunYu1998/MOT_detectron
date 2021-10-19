import torch
from setuptools import find_packages, setup

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 4], "Requires PyTorch >= 1.4"

setup(
    name="mot_detectron",
    version="0.1",
    author="HaojunYu",
    description="MOT",
    packages=find_packages(exclude=("outputs","datasets")),
    python_requires=">=3.6",
    install_requires=['numpy', 'torch', 'pillow', 'detectron2', 'pathspec']
)
