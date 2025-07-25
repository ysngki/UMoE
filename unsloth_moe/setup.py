from setuptools import setup, find_packages

setup(
    name="unsloth_moe",
    version="0.1",
    packages=find_packages(),
    # 添加依赖项
    install_requires=[
        "torch",
        "pandas",
        "pytest",
        "ruff",
    ],
    # # 特殊依赖项需要使用dependency_links
    # dependency_links=[
    #     "git+https://github.com/huggingface/transformers.git@main#egg=transformers",
    # ],
)