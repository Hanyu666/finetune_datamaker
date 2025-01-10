import json
from typing import List, Dict, Tuple
import base64
from pathlib import Path
import openai
from PIL import Image
import io
import httpx
import asyncio
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, api_config):
        self.api_config = api_config
        self.default_prompt = "请详细描述这张图片的内容，包括主要对象、场景、动作和显著特征。"
        
    def _encode_image(self, image_path: str) -> str:
        """将图片转换为base64编码"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def _create_chat_completion(self, messages: List[Dict], max_retries: int = 3, timeout: float = 60.0) -> str:
        """创建聊天完成请求"""
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                async with httpx.AsyncClient() as client:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_config.api_key}"
                    }
                    data = {
                        "model": self.api_config.model_name,
                        "messages": messages,
                        "max_tokens": 300,
                        "stream": False
                    }
                    
                    # 构建完整的API URL
                    base_url = self.api_config.base_url.rstrip('/')
                    api_url = f"{base_url}/v1/chat/completions"
                    
                    response = await client.post(
                        api_url,
                        headers=headers,
                        json=data,
                        timeout=timeout
                    )
                    response.raise_for_status()
                    
                    # 记录原始响应内容
                    raw_content = response.content.decode('utf-8')
                    
                    try:
                        response_data = response.json()
                    except json.JSONDecodeError as e:
                        raise Exception(f"API响应格式错误: {raw_content[:200]}")
                    
                    if "choices" not in response_data or not response_data["choices"]:
                        raise Exception("API响应缺少choices字段")
                        
                    if "message" not in response_data["choices"][0]:
                        raise Exception("API响应缺少message字段")
                        
                    if "content" not in response_data["choices"][0]["message"]:
                        raise Exception("API响应缺少content字段")
                    
                    content = response_data["choices"][0]["message"]["content"]
                    return content.strip()
                    
            except httpx.TimeoutException:
                last_error = "API请求超时"
            except httpx.HTTPStatusError as e:
                error_content = e.response.content.decode('utf-8') if e.response else "未知错误"
                last_error = f"API请求失败: {e.response.status_code}, {error_content}"
                # 如果是认证错误或者其他4xx错误，不再重试
                if e.response.status_code >= 400 and e.response.status_code < 500:
                    raise Exception(last_error)
            except Exception as e:
                last_error = f"API请求异常: {str(e)}"
            
            retry_count += 1
            if retry_count < max_retries:
                # 指数退避重试
                await asyncio.sleep(2 ** retry_count)
        
        raise Exception(f"达到最大重试次数({max_retries})，最后一次错误: {last_error}")
            
    async def process_image(self, image_path: str, prompt: str = None) -> Dict:
        """处理单张图片"""
        try:
            # 验证图片
            image = Image.open(image_path)
            # 转换为base64
            base64_image = self._encode_image(image_path)
            
            # 准备API调用
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or self.default_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            try:
                # 调用API
                description = await self._create_chat_completion(messages)
                return {
                    "success": True,
                    "description": description
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"API调用失败: {str(e)}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"图片处理失败: {str(e)}"
            }
            
    async def process_batch(self, image_paths: List[str], prompt: str = None) -> List[Dict]:
        """批量处理图片"""
        results = []
        errors = []
        
        for idx, image_path in enumerate(image_paths, 1):
            try:
                logger.info(f"处理第 {idx}/{len(image_paths)} 张图片: {image_path}")
                result = await self.process_image(image_path, prompt)
                
                if result["success"]:
                    logger.info(f"第 {idx} 张图片处理成功")
                    results.append({
                        "id": idx,
                        "image_path": image_path,
                        "description": result["description"]
                    })
                else:
                    error_msg = f"第 {idx} 张图片处理失败: {result['error']}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    
            except Exception as e:
                error_msg = f"第 {idx} 张图片处理异常: {str(e)}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
                
        if not results and errors:
            # 如果所有图片都处理失败，抛出异常
            raise Exception("所有图片处理失败:\n" + "\n".join(errors))
            
        if errors:
            # 如果部分图片处理失败，记录警告
            logger.warning(f"部分图片处理失败 ({len(errors)}/{len(image_paths)}):\n" + "\n".join(errors))
            
        logger.info(f"批量处理完成，成功: {len(results)}, 失败: {len(errors)}")
        return results
        
    def save_dataset(self, data: List[Dict], output_path: str) -> Tuple[bool, str]:
        """保存数据集"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True, "数据集保存成功！"
        except Exception as e:
            return False, f"保存失败：{str(e)}" 