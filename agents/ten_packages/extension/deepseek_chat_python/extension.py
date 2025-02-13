#
#
# Agora Real Time Engagement
# Created by Wei Hu in 2024-08.
# Copyright (c) 2024 Agora IO. All rights reserved.
#
#
import asyncio
import json
import traceback
from typing import Iterable

from ten.async_ten_env import AsyncTenEnv
from ten_ai_base.const import CMD_PROPERTY_RESULT, CMD_TOOL_CALL
from ten_ai_base.helper import (
    AsyncEventEmitter,
    get_property_bool,
    get_property_string,
)
from ten_ai_base import AsyncLLMBaseExtension
from ten_ai_base.types import (
    LLMCallCompletionArgs,
    LLMChatCompletionContentPartParam,
    LLMChatCompletionUserMessageParam,
    LLMChatCompletionMessageParam,
    LLMDataCompletionArgs,
    LLMToolMetadata,
    LLMToolResult,
)

from .helper import parse_sentences
from .openai import OpenAIChatGPT, OpenAIChatGPTConfig
from ten import (
    Cmd,
    StatusCode,
    CmdResult,
    Data,
)

CMD_IN_FLUSH = "flush"
CMD_IN_ON_USER_JOINED = "on_user_joined"
CMD_IN_ON_USER_LEFT = "on_user_left"
CMD_OUT_FLUSH = "flush"
DATA_IN_TEXT_DATA_PROPERTY_TEXT = "text"
DATA_IN_TEXT_DATA_PROPERTY_IS_FINAL = "is_final"
DATA_OUT_TEXT_DATA_PROPERTY_TEXT = "text"
DATA_OUT_TEXT_DATA_PROPERTY_TEXT_END_OF_SEGMENT = "end_of_segment"


class OpenAIChatGPTExtension(AsyncLLMBaseExtension):
    def __init__(self, name: str):
        super().__init__(name)
        self.memory = []
        self.memory_cache = []
        self.config = None
        self.client = None
        self.sentence_fragment = ""
        self.tool_task_future: asyncio.Future | None = None
        self.users_count = 0

    async def on_init(self, async_ten_env: AsyncTenEnv) -> None:
        async_ten_env.log_info("on_init")
        await super().on_init(async_ten_env)

    async def on_start(self, async_ten_env: AsyncTenEnv) -> None:
        async_ten_env.log_info("on_start")
        await super().on_start(async_ten_env)

        self.config = await OpenAIChatGPTConfig.create_async(ten_env=async_ten_env)

        # Mandatory properties
        if not self.config.api_key:
            async_ten_env.log_info("API key is missing, exiting on_start")
            return

        # Create instance
        try:
            self.client = OpenAIChatGPT(async_ten_env, self.config)
            async_ten_env.log_info(
                f"initialized with max_tokens: {self.config.max_tokens}, model: {self.config.model}, vendor: {self.config.vendor}"
            )
        except Exception as err:
            async_ten_env.log_info(f"Failed to initialize OpenAIChatGPT: {err}")

    async def on_stop(self, async_ten_env: AsyncTenEnv) -> None:
        async_ten_env.log_info("on_stop")
        await super().on_stop(async_ten_env)

    async def on_deinit(self, async_ten_env: AsyncTenEnv) -> None:
        async_ten_env.log_info("on_deinit")
        await super().on_deinit(async_ten_env)

    async def on_cmd(self, async_ten_env: AsyncTenEnv, cmd: Cmd) -> None:
        cmd_name = cmd.get_name()
        async_ten_env.log_info(f"on_cmd name: {cmd_name}")

        if cmd_name == CMD_IN_FLUSH:
            await self.flush_input_items(async_ten_env)
            await async_ten_env.send_cmd(Cmd.create(CMD_OUT_FLUSH))
            async_ten_env.log_info("on_cmd sent flush")
            status_code, detail = StatusCode.OK, "success"
            cmd_result = CmdResult.create(status_code)
            cmd_result.set_property_string("detail", detail)
            await async_ten_env.return_result(cmd_result, cmd)
        elif cmd_name == CMD_IN_ON_USER_JOINED:
            self.users_count += 1
            # Send greeting when first user joined
            if self.config.greeting and self.users_count == 1:
                self.send_text_output(async_ten_env, self.config.greeting, True)

            status_code, detail = StatusCode.OK, "success"
            cmd_result = CmdResult.create(status_code)
            cmd_result.set_property_string("detail", detail)
            await async_ten_env.return_result(cmd_result, cmd)
        elif cmd_name == CMD_IN_ON_USER_LEFT:
            self.users_count -= 1
            status_code, detail = StatusCode.OK, "success"
            cmd_result = CmdResult.create(status_code)
            cmd_result.set_property_string("detail", detail)
            await async_ten_env.return_result(cmd_result, cmd)
        else:
            await super().on_cmd(async_ten_env, cmd)

    async def on_data(self, async_ten_env: AsyncTenEnv, data: Data) -> None:
        data_name = data.get_name()
        async_ten_env.log_info("[chatgpt_python] on_data name {}".format(data_name))

        # Get the necessary properties
        is_final = get_property_bool(data, "is_final")
        input_text = get_property_string(data, "text")

        if not is_final:
            async_ten_env.log_debug("ignore non-final input")
            return
        if not input_text:
            async_ten_env.log_warn("ignore empty text")
            return

        async_ten_env.log_info(f"OnData input text: [{input_text}]")

        # Start an asynchronous task for handling chat completion
        message = LLMChatCompletionUserMessageParam(role="user", content=input_text)
        await self.queue_input_item(False, messages=[message], input_text=input_text)

    async def on_tools_update(
        self, async_ten_env: AsyncTenEnv, tool: LLMToolMetadata
    ) -> None:
        return await super().on_tools_update(async_ten_env, tool)

    async def on_call_chat_completion(
        self, async_ten_env: AsyncTenEnv, **kargs: LLMCallCompletionArgs
    ) -> any:
        kmessages: LLMChatCompletionUserMessageParam = kargs.get("messages", [])

        async_ten_env.log_info(f"on_call_chat_completion: {kmessages}")
        response = await self.client.get_chat_completions(kmessages, None)
        return response.to_json()

    async def on_data_chat_completion(
        self, async_ten_env: AsyncTenEnv, **kargs: LLMDataCompletionArgs
    ) -> None:
        try:
            messages = []
            tools = None
            no_tool = kargs.get("no_tool", False)

            # Convert input text to message
            input_text = kargs.get("input_text")
            async_ten_env.log_info(f"Processing input text: {input_text}")
            
            if input_text is None:
                async_ten_env.log_error("Input text is None")
                return
                
            if isinstance(input_text, str):
                messages.append({"role": "user", "content": input_text})
            elif isinstance(input_text, list):
                for item in input_text:
                    messages.append(self.message_to_dict(item))
            else:
                async_ten_env.log_error(f"Invalid input text type: {type(input_text)}")
                return

            # Add the messages to memory cache
            self.memory_cache = []
            for m in messages:
                if m.get("content"):  # 只添加有内容的消息
                    self.memory_cache.append(m)
                    self._append_memory(m)
                    async_ten_env.log_info(f"Added new message to memory: {m}")

            # 如果历史记录为空，添加一个空的 assistant 消息
            if not self.memory_cache:
                self.memory_cache.append({"role": "assistant", "content": ""})

            # Create an event for signaling content finished
            content_finished_event = asyncio.Event()

            # Validate message content
            for message in messages:
                if (
                    not isinstance(message.get("content"), str)
                    and not isinstance(message.get("content"), list)
                ):
                    async_ten_env.log_error(
                        f"Invalid message content type: {type(message.get('content'))}"
                    )
                    return

            # Convert tools if needed
            if not no_tool and kargs.get("tools"):
                tools = [self._convert_tools_to_dict(t) for t in kargs["tools"]]

            # Get memory
            memory = []
            if not kargs.get("no_memory", False):
                memory = self.memory
                async_ten_env.log_info(f"Current memory: {memory}")

            # Reset state
            self.sentence_fragment = ""

            # Create a future to track the single tool call task
            self.tool_task_future = None

            # 准备发送给 API 的消息列表
            api_messages = memory + messages
            # 确保消息不重复
            seen = set()
            unique_messages = []
            for m in api_messages:
                msg_key = (m.get("role", ""), m.get("content", ""))
                if msg_key not in seen:
                    seen.add(msg_key)
                    unique_messages.append(m)

            # Log the content being sent to OpenAI API
            async_ten_env.log_info(
                f"Sending to OpenAI API - Messages: {unique_messages}, Tools: {tools}"
            )

            # Create an async listener to handle tool calls and content updates
            async def handle_tool_call(tool_call):
                self.tool_task_future = asyncio.get_event_loop().create_future()
                async_ten_env.log_info(f"tool_call: {tool_call}")
                for tool in self.available_tools:
                    if tool_call["function"]["name"] == tool.name:
                        cmd: Cmd = Cmd.create(CMD_TOOL_CALL)
                        cmd.set_property_string("name", tool.name)
                        cmd.set_property_from_json(
                            "arguments", tool_call["function"]["arguments"]
                        )

                        # Send the command and handle the result through the future
                        [result, _] = await async_ten_env.send_cmd(cmd)
                        if result.get_status_code() == StatusCode.OK:
                            tool_result: LLMToolResult = json.loads(
                                result.get_property_to_json(CMD_PROPERTY_RESULT)
                            )

                            async_ten_env.log_info(f"tool_result: {tool_result}")

                            # Handle tool result
                            if tool_result["type"] == "llmresult":
                                result_content = tool_result["content"]
                                if isinstance(result_content, str):
                                    tool_message = {
                                        "role": "assistant",
                                        "tool_calls": [tool_call],
                                    }
                                    new_message = {
                                        "role": "tool",
                                        "content": result_content,
                                        "tool_call_id": tool_call["id"],
                                    }
                                    await self.queue_input_item(
                                        True, messages=[tool_message, new_message], no_tool=True
                                    )
                                else:
                                    async_ten_env.log_error(
                                        f"Unknown tool result content: {result_content}"
                                    )
                            elif tool_result["type"] == "requery":
                                self.memory_cache.pop()
                                result_content = tool_result["content"]
                                new_message = {
                                    "role": "user",
                                    "content": self._convert_to_content_parts(
                                        messages[0]["content"]
                                    ),
                                }
                                new_message["content"] = new_message[
                                    "content"
                                ] + self._convert_to_content_parts(result_content)
                                await self.queue_input_item(
                                    True, messages=[new_message], no_tool=True
                                )
                            else:
                                async_ten_env.log_error(
                                    f"Unknown tool result type: {tool_result}"
                                )
                        else:
                            async_ten_env.log_error("Tool call failed")
                self.tool_task_future.set_result(None)

            async def handle_content_update(content: str):
                # Append the content to the last assistant message
                found_assistant = False
                for item in reversed(self.memory_cache):
                    if item.get("role") == "assistant":
                        found_assistant = True
                        if item["content"] is None:
                            item["content"] = content
                        else:
                            item["content"] = item["content"] + content
                        break
                
                if not found_assistant:
                    async_ten_env.log_info("Creating new assistant message in memory_cache")
                    # Add a new assistant message
                    self.memory_cache.append({
                        "role": "assistant",
                        "content": content
                    })

                try:
                    sentences, self.sentence_fragment = parse_sentences(
                        self.sentence_fragment, content
                    )
                    for s in sentences:
                        self.send_text_output(async_ten_env, s, False)
                except Exception as e:
                    async_ten_env.log_error(
                        f"Error in handle_content_update: {str(e)}, content: {content}, fragment: {self.sentence_fragment}"
                    )
                    raise

            async def handle_content_finished(_: str):
                # Wait for the single tool task to complete (if any)
                if self.tool_task_future:
                    await self.tool_task_future
                content_finished_event.set()

            listener = AsyncEventEmitter()
            listener.on("tool_call", handle_tool_call)
            listener.on("content_update", handle_content_update)
            listener.on("content_finished", handle_content_finished)

            # Make an async API call to get chat completions
            try:
                await self.client.get_chat_completions_stream(
                    unique_messages, tools, listener
                )
            except Exception as e:
                async_ten_env.log_error(
                    f"Error during stream: {str(e)}, type: {type(e)}, memory_cache: {self.memory_cache}"
                )
                raise

            # Wait for the content to be finished
            try:
                await asyncio.wait_for(content_finished_event.wait(), timeout=30.0)
                # Add the final assistant message to memory
                for m in self.memory_cache:
                    if m.get("role") == "assistant":
                        self._append_memory(m)
                        async_ten_env.log_info(f"Added assistant response to memory: {m}")
            except asyncio.TimeoutError:
                async_ten_env.log_error(f"Content finished event timeout after 30s")
                raise
            except Exception as e:
                async_ten_env.log_error(
                    f"Error waiting for content: {str(e)}, type: {type(e)}"
                )
                raise

            async_ten_env.log_info(
                f"Chat completion finished for input text: {messages}, memory_cache: {self.memory_cache}"
            )

        except Exception as e:
            async_ten_env.log_error(
                f"Error in chat_completion: {str(e)} for input text: {messages}"
            )
            raise
        finally:
            # Send an empty sentence to mark the end of the segment
            self.send_text_output(async_ten_env, "", True)

    def _convert_to_content_parts(
        self, content: Iterable[LLMChatCompletionContentPartParam]
    ):
        content_parts = []

        if isinstance(content, str):
            content_parts.append({"type": "text", "text": content})
        else:
            for part in content:
                content_parts.append(part)
        return content_parts

    def _convert_tools_to_dict(self, tool: LLMToolMetadata):
        json_dict = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            "strict": True,
        }

        for param in tool.parameters:
            json_dict["function"]["parameters"]["properties"][param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.required:
                json_dict["function"]["parameters"]["required"].append(param.name)

        return json_dict

    def message_to_dict(self, message: LLMChatCompletionMessageParam):
        if message.get("content") is not None:
            if isinstance(message["content"], str):
                message["content"] = str(message["content"])
            else:
                message["content"] = list(message["content"])
        return message

    def _append_memory(self, message):
        """Append a message to memory, respecting the max memory length."""
        if not message or not isinstance(message, dict):
            return

        # 不添加空内容的消息
        if not message.get("content"):
            return
        
        # 添加新消息
        self.memory.append(message)
        
        # 如果超过最大长度，移除最早的消息
        while len(self.memory) > self.config.max_memory_length:
            self.memory.pop(0)
