import json
from typing import List, Dict
import openai
from pathlib import Path
import httpx
import logging
import asyncio

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, api_config):
        self.api_config = api_config
        self._load_default_prompts()
        
    def _load_default_prompts(self):
        """加载默认的agent提示词"""
        self.text_analysis_prompt = """你是文本分析专家。
你的任务是分析输入的文本，在接近1000个token的位置找到合适的语义断点，这个断点应该尽量保持段落或语义的完整性。

要求：
1. 寻找最接近1000 token处的语义完整位置
2. 优先在段落结束处断开
3. 返回从开始到断点的完整文本
4. 关注语义连贯性，不要在句子中间断开

直接返回这段完整的文本。"""

        self.title_generation_prompt = """你是标题生成专家。
为文本生成简短的instruction，要求：
1. 长度控制在10个字以内
2. 直接概括文本核心主题
3. 避免过度解释或分析

直接返回标题文本。"""

        self.format_prompt = """你是格式化专家。
请将提供的标题和原文按以下格式组织成JSON（直接返回 JSON，不要包含任何其他标记）：
{
    "instruction": "标题",
    "input": "",
    "output": "原文"
}

注意：
1. 严格按照格式输出
2. 保持原文完整
3. 只返回纯 JSON 字符串，不要包含 markdown 代码块标记
4. 确保 JSON 中的换行使用 \n
5. 不要添加任何额外的说明或注释，只返回JSON"""
        
    def update_prompts(self, text_analysis: str, title_generation: str, format_prompt: str):
        """更新agent提示词"""
        self.text_analysis_prompt = text_analysis
        self.title_generation_prompt = title_generation
        self.format_prompt = format_prompt

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
                        "stream": False
                    }
                    
                    # 构建完整的API URL
                    base_url = self.api_config.base_url.rstrip('/')
                    api_url = f"{base_url}/v1/chat/completions"
                    logger.debug(f"Making request to: {api_url}")
                    
                    response = await client.post(
                        api_url,
                        headers=headers,
                        json=data,
                        timeout=timeout
                    )
                    response.raise_for_status()
                    
                    # 记录原始响应内容
                    raw_content = response.content.decode('utf-8')
                    logger.debug(f"Raw API Response: {raw_content}")
                    
                    try:
                        response_data = response.json()
                    except json.JSONDecodeError as e:
                        logger.error(f"API响应解析失败: {raw_content}")
                        raise Exception(f"API响应格式错误，请检查API地址是否正确: {raw_content[:200]}")
                    
                    if "choices" not in response_data or not response_data["choices"]:
                        raise Exception("API响应缺少choices字段")
                        
                    if "message" not in response_data["choices"][0]:
                        raise Exception("API响应缺少message字段")
                        
                    if "content" not in response_data["choices"][0]["message"]:
                        raise Exception("API响应缺少content字段")
                    
                    content = response_data["choices"][0]["message"]["content"]
                    logger.debug(f"Processed API Response: {content}")
                    return content.strip()
                    
            except httpx.TimeoutException:
                last_error = "API请求超时"
                logger.warning(f"API请求超时，重试次数: {retry_count + 1}/{max_retries}")
            except httpx.HTTPStatusError as e:
                error_content = e.response.content.decode('utf-8') if e.response else "未知错误"
                last_error = f"API请求失败: {e.response.status_code}, {error_content}"
                logger.error(f"HTTP错误: {error_content}")
                # 如果是认证错误或者其他4xx错误，不再重试
                if e.response.status_code >= 400 and e.response.status_code < 500:
                    raise Exception(last_error)
            except Exception as e:
                last_error = f"API请求异常: {str(e)}"
                logger.error(f"请求异常: {str(e)}", exc_info=True)
            
            retry_count += 1
            if retry_count < max_retries:
                # 指数退避重试
                await asyncio.sleep(2 ** retry_count)
        
        raise Exception(f"达到最大重试次数({max_retries})，最后一次错误: {last_error}")

    async def process_text(self, text: str) -> Dict:
        """处理文本并返回结果"""
        try:
            all_results = []
            remaining_text = text
            
            while remaining_text.strip():
                logger.info("开始处理新的文本片段")
                
                # 1. 文本分析
                logger.info("开始文本分析")
                messages = [
                    {"role": "system", "content": self.text_analysis_prompt},
                    {"role": "user", "content": remaining_text}
                ]
                try:
                    processed_text = await self._create_chat_completion(messages)
                    logger.debug(f"文本分析结果: {processed_text[:100]}...")
                    # 更新剩余文本
                    remaining_text = remaining_text[len(processed_text):].strip()
                except Exception as e:
                    logger.error(f"文本分析失败: {str(e)}")
                    return {
                        "success": False,
                        "error": f"文本分析失败: {str(e)}"
                    }

                # 2. 生成标题
                logger.info("开始生成标题")
                messages = [
                    {"role": "system", "content": self.title_generation_prompt},
                    {"role": "user", "content": processed_text}
                ]
                try:
                    title = await self._create_chat_completion(messages)
                    logger.debug(f"生成的标题: {title}")
                except Exception as e:
                    logger.error(f"标题生成失败: {str(e)}")
                    return {
                        "success": False,
                        "error": f"标题生成失败: {str(e)}"
                    }

                # 3. 格式化
                logger.info("开始格式化")
                messages = [
                    {"role": "system", "content": self.format_prompt},
                    {"role": "user", "content": f"标题：{title}\n原文：{processed_text}"}
                ]
                try:
                    formatted_text = await self._create_chat_completion(messages)
                    logger.debug(f"格式化结果: {formatted_text}")
                    
                    # 尝试修复可能的JSON格式问题
                    formatted_text = formatted_text.strip()
                    if not formatted_text.startswith('{'):
                        formatted_text = formatted_text[formatted_text.find('{'):]
                    if not formatted_text.endswith('}'):
                        formatted_text = formatted_text[:formatted_text.rfind('}')+1]
                    
                    formatted_data = json.loads(formatted_text)
                    logger.info("JSON解析成功")
                    all_results.append(formatted_data)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析失败: {str(e)}\n原始文本: {formatted_text}")
                    # 构造一个基本的JSON结构
                    all_results.append({
                        "instruction": title,
                        "input": "",
                        "output": processed_text
                    })
                except Exception as e:
                    logger.error(f"格式化失败: {str(e)}")
                    return {
                        "success": False,
                        "error": f"格式化失败: {str(e)}"
                    }
            
            return {
                "success": True,
                "data": all_results
            }
                
        except Exception as e:
            logger.error(f"处理失败: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
            
    def save_dataset(self, data: List[Dict], output_path: str):
        """保存数据集"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"数据集保存成功: {output_path}")
            return True, "数据集保存成功！"
        except Exception as e:
            logger.error(f"保存失败: {str(e)}", exc_info=True)
            return False, f"保存失败：{str(e)}" 