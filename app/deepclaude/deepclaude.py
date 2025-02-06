import json
import time
import asyncio
from typing import AsyncGenerator
from app.utils.logger import logger
from app.clients import DeepSeekClient, ClaudeClient


class DeepClaude:
    def __init__(self, deepseek_api_key: str, claude_api_key: str, 
                 deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions", 
                 claude_api_url: str = "https://api.anthropic.com/v1/messages",
                 claude_provider: str = "anthropic",
                 is_origin_reasoning: bool = True):
        self.deepseek_client = DeepSeekClient(deepseek_api_key, deepseek_api_url)
        self.claude_client = ClaudeClient(claude_api_key, claude_api_url, claude_provider)
        self.is_origin_reasoning = is_origin_reasoning
    
    async def chat_completions_with_stream(self, messages: list, 
                                         deepseek_model: str = "deepseek-reasoner",
                                         claude_model: str = "claude-3-5-sonnet-20241022") -> AsyncGenerator[bytes, None]:
        chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
        created_time = int(time.time())

        output_queue = asyncio.Queue()
        claude_queue = asyncio.Queue()
        reasoning_content = []
        
        async def process_deepseek():
            logger.info(f"[DeepSeek] 开始流式处理，使用模型: {deepseek_model}")
            try:
                async for content_type, content in self.deepseek_client.stream_chat(messages, deepseek_model, self.is_origin_reasoning):
                    logger.debug(f"[DeepSeek] 收到消息 - 类型: {content_type}, 内容预览: {content[:50] if content else '无内容'}")
                    if content_type == "reasoning":
                        reasoning_content.append(content)
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": deepseek_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "reasoning_content": content,
                                    "content": ""
                                }
                            }]
                        }
                        await output_queue.put(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
                    elif content_type == "content":
                        logger.info(f"[DeepSeek] 推理结束，累计推理内容长度: {len(''.join(reasoning_content))}")
                        await claude_queue.put("".join(reasoning_content))
                        break
            except Exception as e:
                logger.error(f"[DeepSeek] 流处理错误: {e}")
                await claude_queue.put("")
            logger.info("[DeepSeek] 任务结束")
            await output_queue.put(None)
        
        async def process_claude():
            try:
                logger.info("[Claude] 等待 DeepSeek 推理结果...")
                reasoning = await claude_queue.get()
                logger.debug(f"[Claude] 获取到的推理内容长度: {len(reasoning) if reasoning else 0}")
                if not reasoning:
                    logger.warning("[Claude] DeepSeek 推理内容为空，使用默认值")
                    reasoning = "获取推理内容失败"
                
                claude_messages = messages.copy()
                claude_messages.append({
                    "role": "assistant",
                    "content": f"Here's my reasoning process:\n{reasoning}\n\nBased on this reasoning, I will now provide my response:"
                })
                
                claude_messages = [msg for msg in claude_messages if msg.get("role", "") != "system"]
                
                async for content_type, content in self.claude_client.stream_chat(claude_messages, claude_model):
                    logger.debug(f"[Claude] 收到消息 - 类型: {content_type}, 内容预览: {content[:50] if content else '无内容'}")
                    if content_type == "answer":
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": claude_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": content
                                }
                            }]
                        }
                        await output_queue.put(f"data: {json.dumps(response)}\n\n".encode('utf-8'))
            except Exception as e:
                logger.error(f"[Claude] 处理流错误: {e}")
            await output_queue.put(None)
        
        deepseek_task = asyncio.create_task(process_deepseek())
        claude_task = asyncio.create_task(process_claude())
        
        finished_tasks = 0
        while finished_tasks < 2:
            item = await output_queue.get()
            if item is None:
                finished_tasks += 1
            else:
                yield item
        
        yield b'data: [DONE]\n\n'
