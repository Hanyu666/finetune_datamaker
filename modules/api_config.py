import openai
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class APIConfig:
    def __init__(self):
        self.api_key: Optional[str] = None
        self.base_url: Optional[str] = None
        self.model_name: Optional[str] = None
        
    def set_config(self, api_key: str, base_url: str, model_name: str) -> None:
        """设置API配置"""
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')  # 移除末尾的斜杠
        self.model_name = model_name
        
        # 配置OpenAI客户端
        openai.api_key = api_key
        if base_url:
            # 确保正确的URL格式
            base_url = base_url.rstrip('/')
            openai.base_url = f"{base_url}/v1"
            logger.debug(f"OpenAI base_url set to: {openai.base_url}")
            
    def test_connection(self) -> Tuple[bool, str]:
        """测试API连接"""
        if not all([self.api_key, self.model_name]):
            return False, "请先完成API配置"
        
        try:
            # 记录当前配置
            logger.debug(f"Testing API with model: {self.model_name}")
            logger.debug(f"Current base_url: {openai.base_url}")
            
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=f"{self.base_url.rstrip('/')}/v1"
            )
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True, "API连接测试成功！"
        except Exception as e:
            error_msg = f"API连接测试失败：{str(e)}"
            logger.error(error_msg)
            return False, error_msg
            
    def get_config(self) -> Dict[str, str]:
        """获取当前配置"""
        return {
            "api_key": self.api_key or "",
            "base_url": self.base_url or "",
            "model_name": self.model_name or ""
        } 