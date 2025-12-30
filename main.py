import yaml

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

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
    return f"{city}总是阳光明媚！"


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户ID检索用户信息。"""
    user_id = runtime.context.user_id
    return "佛罗里达" if user_id == "1" else "旧金山"


# 定义响应模式
@dataclass
class ResponseFormat:
    """智能体的响应模式。"""

    punny_response: str  # 带双关语的响应
    weather_conditions: str | None = None  # 天气状况（可选）


# model = ChatOpenAI(model=MODEL_NAME, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


# 从tools.yaml加载配置
def load_config(config_path):
    """从YAML文件加载配置"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


tool_config = load_config("./tools.yaml")  # 加载配置
model = ChatOpenAI(**tool_config["llm"])  # 创建模型实例
advance_model = ChatOpenAI(**tool_config["advance_llm"])  # 创建高级模型实例


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
    # `thread_id`是给定对话的唯一标识符。
    config = {"configurable": {"thread_id": "1"}}
    response = agent.invoke(  # 调用智能体
        {"messages": [{"role": "user", "content": "外面天气怎么样？"}]},  # 用户消息
        config=config,  # 配置（包含thread_id）
        context=Context(user_id="1"),  # 上下文（包含用户ID）
    )
    print(response["structured_response"])  # 打印结构化响应
    # 注意，我们可以使用相同的`thread_id`继续对话。
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "thank you!"}]}, config=config, context=Context(user_id="1")
    )
    print(response["structured_response"])


if __name__ == "__main__":
    main()
