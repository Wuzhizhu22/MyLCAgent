import yaml

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langchain_core.callbacks import get_usage_metadata_callback

from dataclasses import dataclass

SYSTEM_PROMPT = """你是一位专业的天气预报员，说话时喜欢用双关语。

你可以使用两个工具：

- get_weather_for_location：使用此工具获取特定位置的天气
- get_user_location：使用此工具获取用户的位置

如果用户向你询问天气，确保你知道位置。如果从问题中可以看出他们指的是他们所在的位置，请使用get_user_location工具查找他们的位置。"""


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


tool_config = load_config("./llm.yaml")  # 加载配置
model = ChatOpenAI(**tool_config["llm"])  # 创建模型实例


def main():
    checkpointer = InMemorySaver()  # 创建内存检查点保存器
    agent = create_agent(  # 创建智能体
        model=model,  # 传入模型
        system_prompt=SYSTEM_PROMPT,  # 传入系统提示词
        tools=[get_user_location, get_weather_for_location],  # 传入可用工具
        context_schema=Context,  # 传入上下文模式
        response_format=ResponseFormat,  # 传入响应模式
        checkpointer=checkpointer,  # 传入检查点保存器
    )
    config = {"configurable": {"thread_id": "1"}}

    print("=== 开始流式输出 ===\n")

    # 使用 stream_mode=["updates", "custom"] 同时获取更新和自定义流式输出
    for stream_mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": "天气如何呢?"}]},
        stream_mode=["updates", "custom"],
        config=config,
        context=Context(user_id="1"),  # 上下文（包含用户ID）
    ):
        print(f"📡 流模式: {stream_mode}")

        if stream_mode == "custom":
            # 自定义流式输出（来自工具内部的 get_stream_writer）
            print(f"  🎯 {chunk}")
        elif stream_mode == "updates":
            # 更新模式（步骤信息）
            for step, data in chunk.items():
                print(f"📍 步骤: {step}")

                # 处理消息内容
                messages = data.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    content_blocks = getattr(last_message, "content_blocks", None)

                    if content_blocks:
                        for block in content_blocks:
                            if block.get("type") == "tool_call":
                                print(f"  🛠️  调用工具: {block.get('name')}")
                                print(f"  📝 参数: {block.get('args')}")
                            elif block.get("type") == "text":
                                print(f"  💬 内容: {block.get('text')}")
                    else:
                        print(f"  📄 消息内容: {last_message.content}")

                print()  # 空行分隔

    print("\n=== 流式输出完成 ===")


if __name__ == "__main__":
    main()
