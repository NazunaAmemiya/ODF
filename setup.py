import os
from setuptools import setup, find_packages

def parse_requirements(filename):
    """Đọc danh sách thư viện từ file requirements.txt."""
    with open(filename, 'r', encoding='utf-8') as f:
        # Bỏ qua các dòng trống và các dòng comment (bắt đầu bằng #)
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="mosquito_cv",
    version="0.1.0",
    author="Your Name or Team",
    author_email="your.email@example.com",
    description="A Modular Computer Vision Framework for Mosquito Detection and Segmentation",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    
    # Chỉ định thư mục chứa mã nguồn cốt lõi
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    python_requires=">=3.8",
    # Tự động đọc danh sách từ file requirements.txt
    install_requires=parse_requirements("requirements.txt"),
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
)