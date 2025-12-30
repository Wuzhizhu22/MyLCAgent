"""
优化后的 main.py - 使用 LangChain 消息对象
"""

import yaml
import argparse

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from dataclasses import dataclass
from typing import Optional
import uuid

# 使用 SystemMessage 对象代替字符串
SYSTEM_MESSAGE = SystemMessage(
    content="""你是一位专业的天气预报员，说话时喜欢用双关语。

你可以使用两个工具：

- get_weather_for_location：使用此工具获取特定位置的天气
- get_user_location：使用此工具获取用户的位置

如果用户向你询问天气，确保你知道位置。如果从问题中可以看出他们指的是他们所在的位置，请使用get_user_location工具查找他们的位置。""",
    id="system_001",  # 如果未来添加多系统提示切换等功能可用
)


@dataclass
class Context:
    """自定义运行时上下文模式。"""

    user_id: str


@tool
def get_weather_for_location(city: str) -> str:
    """获取给定城市的天气。"""
    writer = get_stream_writer()
    writer(f"🔍 正在查询城市: {city}")
    writer(f"📊 获取到城市数据: {city}")
    return f"{city}总是阳光明媚！"


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户ID检索用户信息。"""
    writer = get_stream_writer()
    user_id = runtime.context.user_id
    writer(f"👤 正在查找用户ID: {user_id}")
    location = "佛罗里达" if user_id == "1" else "旧金山"
    writer(f"📍 找到用户位置: {location}")
    return location


# 定义响应模式
@dataclass
class ResponseFormat:
    punny_response: str
    weather_conditions: str | None = None

    def __str__(self):
        result = f"回答：{self.punny_response}"
        if self.weather_conditions:
            result += f"\n天气状况：{self.weather_conditions}"
        return result


# 从llm.yaml加载配置
def load_config(config_path):
    """从YAML文件加载配置"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_human_message(content: str, user_id: str = None) -> HumanMessage:
    """创建带元数据的人类消息"""
    return HumanMessage(
        content=content, name=f"user_{user_id}" if user_id else "user", id=f"msg_{uuid.uuid4().hex[:8]}"
    )


def process_message_blocks(message: AIMessage) -> dict:
    """处理 AIMessage 的 content_blocks，提取不同类型的内容"""
    result = {"tool_calls": [], "text_content": [], "reasoning": None}

    # 直接访问 content_blocks（不需要 getattr）
    content_blocks = message.content_blocks

    if content_blocks:
        for block in content_blocks:
            block_type = block.get("type")

            if block_type == "tool_call":
                result["tool_calls"].append(
                    {"name": block.get("name"), "args": block.get("args"), "id": block.get("id")}
                )
            elif block_type == "text":
                result["text_content"].append(block.get("text"))
            elif block_type == "reasoning":
                # 支持推理过程（如 OpenAI o1 模型）
                result["reasoning"] = block.get("summary", [])

    return result


tool_config = load_config("./llm.yaml")
model = ChatOpenAI(**tool_config["llm"])


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="天气查询智能体")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stream", "invoke"],
        default="invoke",
        help="运行模式：stream（流式输出）或 invoke（一次性输出）",
    )
    args = parser.parse_args()

    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_MESSAGE.content,  # 从 SystemMessage 提取内容
        tools=[get_user_location, get_weather_for_location],
        context_schema=Context,
        response_format=ResponseFormat,
        checkpointer=checkpointer,
    )

    # 创建 usage metadata callback handler
    callback = UsageMetadataCallbackHandler()
    config_with_callback = {"configurable": {"thread_id": "1"}, "callbacks": [callback]}

    # 使用 HumanMessage 对象
    user_message = create_human_message("天气如何呢?", user_id="1")

    if args.mode == "stream":
        # 流式输出模式
        print("=== 开始流式输出 ===\n")

        for stream_mode, chunk in agent.stream(
            {"messages": [user_message]},
            stream_mode=["updates", "custom"],
            config=config_with_callback,
            context=Context(user_id="1"),
        ):
            print(f"📡 流模式: {stream_mode}")

            if stream_mode == "custom":
                print(f"  🎯 {chunk}")
            elif stream_mode == "updates":
                for step, data in chunk.items():
                    print(f"📍 步骤: {step}")

                    messages = data.get("messages", [])
                    if messages:
                        last_message = messages[-1]

                        # 使用优化的消息处理函数
                        if isinstance(last_message, AIMessage):
                            blocks = process_message_blocks(last_message)

                            # 显示工具调用
                            for tool_call in blocks["tool_calls"]:
                                print(f"  🛠️  调用工具: {tool_call['name']}")
                                print(f"  📝 参数: {tool_call['args']}")

                            # 显示文本内容
                            for text in blocks["text_content"]:
                                print(f"  💬 内容: {text}")

                            # 显示推理过程（如果有）
                            if blocks["reasoning"]:
                                print(f"  🧠 推理过程:")
                                for summary in blocks["reasoning"]:
                                    print(f"    - {summary.get('text', '')}")
                        else:
                            # 其他类型的消息
                            print(f"  📄 消息类型: {type(last_message).__name__}")
                            print(f"  📄 消息内容: {last_message.content}")

                    print()

        print("\n=== 流式输出完成 ===")
    else:
        # invoke 模式
        print("=== 使用 invoke 模式 ===\n")

        result = agent.invoke(
            {"messages": [user_message]},
            config=config_with_callback,
            context=Context(user_id="1"),
        )

        print(result["structured_response"])
        print("\n=== invoke 完成 ===")

    # 打印 Token 使用统计
    print("\n📊 Token 使用统计:")
    if callback.usage_metadata:
        for model_name, metadata in callback.usage_metadata.items():
            print(f"  模型: {model_name}")
            print(f"    输入 Tokens: {metadata.get('input_tokens', 0)}")
            print(f"    输出 Tokens: {metadata.get('output_tokens', 0)}")
            print(f"    总计 Tokens: {metadata.get('total_tokens', 0)}")
    else:
        print("  未获取到 token 使用统计")
    # print(f"  完整 metadata: {callback.usage_metadata}")


if __name__ == "__main__":
    main()
