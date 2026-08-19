"""
setup.py — Build and install the alphaz0_cpp C++ extension module.

Usage:
    pip install .          # build + install
    pip install -e .       # editable install (re-links on import)
    python setup.py build_ext --inplace   # build into current directory
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """A setuptools Extension that triggers a CMake build."""
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    """Custom build_ext that invokes CMake."""

    def build_extension(self, ext):
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        build_dir = Path(self.build_temp).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)

        cfg = "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        # Tell CMake where pybind11 is installed
        try:
            import pybind11
            cmake_args.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")
        except ImportError:
            pass

        build_args = ["--config", cfg]

        # Multi-core build
        if hasattr(os, "cpu_count"):
            build_args += ["-j", str(os.cpu_count())]

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_dir,
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_dir,
        )


setup(
    name="alphaz0",
    version="2.0.0",
    author="Anshuman Pandey",
    description="AlphaZ0 — Neural Chess Engine with C++ core",
    long_description=open("../README.md", encoding="utf-8").read()
        if os.path.exists("../README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension("alphaz0_cpp")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "fastapi",
        "uvicorn[standard]",
        "pybind11",
    ],
)
