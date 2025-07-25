# 在 qwen3_moe_fused/__init__.py 中添加
from .interface import grouped_gemm

# 如果有其他想要导出的类或函数也可以添加
__all__ = ['grouped_gemm']