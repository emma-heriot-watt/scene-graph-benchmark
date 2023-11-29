# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
from pathlib import Path

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]


def get_build_extension():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension


def get_extensions():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME
    from torch.utils.cpp_extension import CppExtension
    from torch.utils.cpp_extension import CUDAExtension

    this_dir = Path(__file__).parent

    extensions_dir = this_dir.joinpath("maskrcnn_benchmark", "csrc")
    main_file = list(extensions_dir.glob("*.cpp"))
    source_cpu = list(extensions_dir.glob("cpu/*.cpp"))
    source_cuda = list(extensions_dir.glob("cuda/*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [str(s.relative_to(this_dir)) for s in sources]
    include_dirs = [str(extensions_dir)]

    ext_modules = [
        extension(
            "maskrcnn_benchmark._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="maskrcnn_benchmark",
    version="0.1",
    author="fmassa",
    url="https://github.com/facebookresearch/maskrcnn-benchmark",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": get_build_extension()},
)
