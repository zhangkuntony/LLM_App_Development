# AI助手配置模块
from openai import OpenAI
try:
    from config import BASE_URL, API_KEY, MODEL_NAME  # 从根目录导入配置
    print("✅ 成功从根目录导入配置")
except ImportError:
    print("❌ 无法从根目录导入配置，请检查文件路径")

# 创建全局客户端实例
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 导出配置常量
__all__ = ['client', 'MODEL_NAME', 'BASE_URL', 'API_KEY']