# LLM配置模块
from openai import OpenAI
from config import BASE_URL, API_KEY, MODEL_NAME  # 从根目录导入配置

# 创建全局客户端实例
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 导出配置常量
__all__ = ['client', 'MODEL_NAME', 'BASE_URL', 'API_KEY']