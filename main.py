import yaml
import argparse
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool, ToolRuntime
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer

# --- 配置与常量 ---
SYSTEM_MESSAGE = SystemMessage(
    content="""你是一位专业的天气预报员，说话时喜欢用双关语。
    如果用户询问天气，请确保知道位置。若指代当前位置，请先获取位置再查询。
    请严格按照 ReAct 格式输出：<question>, <thought>, <action>, <observation>, <final_answer>。""",
    id="system_001",
)


@dataclass
class Context:
    user_id: str


class WeatherQuery(BaseModel):
    city: str = Field(description="城市名称")

    @field_validator("city")
    def city_must_not_be_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError("城市名称不能为空")
        return v.strip()


@dataclass
class ResponseFormat:
    question: str
    thought: str
    action: str
    observation: str
    final_answer: str


# --- 工具定义 ---
@tool(
    "get_weather_for_location",
    description="获取指定城市的天气信息。当用户询问天气或需要天气数据时使用此工具，返回指定城市的当前天气状况。",
)
def get_weather_for_location(query: WeatherQuery) -> str:
    """获取指定城市的天气信息"""
    writer = get_stream_writer()
    city = query.city.strip()
    writer(f"🔍 正在查询: {city}")
    return f"{city}总是阳光明媚，真是‘晴’深似海！"


@tool(
    "get_user_location",
    description="根据用户ID获取用户位置信息。当需要知道用户位置时使用此工具，根据运行时上下文中的用户ID返回对应的位置信息。",
)
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """获取用户位置"""
    writer = get_stream_writer()
    location = "佛罗里达" if runtime.context.user_id == "1" else "旧金山"
    writer(f"📍 找到位置: {location}")
    return location


# --- 打印辅助函数 (合并简化) ---
def print_react_step(step_type: str, content: str, tool_args: dict = None) -> None:
    """统一的 ReAct 步骤打印函数"""
    styles = {
        "question": ("❓", "Question"),
        "thought": ("💭", "Thought"),
        "action": ("🔧", "Action"),
        "observation": ("🔍", "Observation"),
        "final_answer": ("✅", "Final Answer"),
    }
    icon, label = styles.get(step_type.lower(), ("📄", step_type.title()))

    # 解码 Unicode 转义序列
    if content and "\\u" in content:
        try:
            content = content.encode("utf-8").decode("unicode_escape")
        except:
            pass  # 如果解码失败，保持原样

    if step_type == "action" and tool_args:
        args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items())
        print(f"\n{icon} {label}: {content}({args_str})")
    else:
        print(f"\n{icon} {label}: {content}")


def print_token_usage(callback: UsageMetadataCallbackHandler) -> None:
    """统计 Token 使用"""
    print(f"\n📊 Token 统计:")
    for model, meta in (callback.usage_metadata or {}).items():
        print(
            f"   {model} -> In: {meta.get('input_tokens')} | Out: {meta.get('output_tokens')} | Total: {meta.get('total_tokens')}"
        )


# --- 核心逻辑解析器 ---
def dispatch_react_elements(message):
    """解析消息内容并分发给打印函数"""
    if isinstance(message, ToolMessage):
        if not message.content.startswith("Returning structured response"):
            print_react_step("observation", message.content)
        return

    if not isinstance(message, AIMessage):
        return

    # 处理思维链或结构化工具调用
    if hasattr(message, "content_blocks") and message.content_blocks:
        for block in message.content_blocks:
            b_type = block.get("type")
            if b_type == "reasoning":
                summary = " ".join([item.get("text", "") for item in block.get("summary", [])])
                print_react_step("thought", summary)
            elif b_type == "tool_call":
                if block.get("name") == "ResponseFormat":
                    args = block.get("args", {})
                    for field in ["question", "thought", "final_answer"]:
                        if args.get(field):
                            print_react_step(field, args[field])
                else:
                    print_react_step("action", block.get("name"), block.get("args"))
            elif b_type == "text" and block.get("text"):
                print_react_step("info", block["text"])


# --- 执行模式 ---
def run_agent(agent, user_input, config, context, mode="invoke"):
    print(f"\n🤔 智能体正在思考 ({mode})...\n")
    user_msg = HumanMessage(content=user_input, name=f"user_{context.user_id}")

    if mode == "stream":
        for stream_mode, chunk in agent.stream(
            {"messages": [user_msg]}, stream_mode=["updates", "custom"], config=config, context=context
        ):
            if stream_mode == "custom":
                print(f"🎯 {chunk}")
            elif stream_mode == "updates":
                for data in chunk.values():
                    if not data:
                        continue
                    if "structured_response" in data:
                        sr = data["structured_response"]
                        for f in ["question", "thought", "final_answer"]:
                            if hasattr(sr, f):
                                print_react_step(f, getattr(sr, f))
                    else:
                        for m in data.get("messages", []):
                            dispatch_react_elements(m)
    else:
        result = agent.invoke({"messages": [user_msg]}, config=config, context=context)
        for m in result.get("messages", []):
            dispatch_react_elements(m)


# --- 主程序 ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation", action="store_true")
    parser.add_argument("--output-mode", choices=["stream", "invoke"], default="invoke")
    parser.add_argument("--show-tokens", action="store_true")
    args = parser.parse_args()

    with open("./llm.yaml", "r") as f:
        llm_config = yaml.safe_load(f)

    model = ChatOpenAI(**llm_config["llm"])
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_MESSAGE.content,
        tools=[get_user_location, get_weather_for_location],
        context_schema=Context,
        response_format=ResponseFormat,
        checkpointer=InMemorySaver(),
        middleware=[SummarizationMiddleware(model=model, trigger=("tokens", 40000), keep=("messages", 20))],
    )

    token_cb = UsageMetadataCallbackHandler()
    config = {"configurable": {"thread_id": "1"}, "callbacks": [token_cb]}
    ctx = Context(user_id="1")

    if args.conversation:
        print("💡 输入 'exit' 退出对话")
        while True:
            inp = input("\n👤 用户: ").strip()
            if inp.lower() in ["exit", "quit", "退出"]:
                break
            if not inp:
                continue
            run_agent(agent, inp, config, ctx, args.output_mode)
            if args.show_tokens:
                print_token_usage(token_cb)
    else:
        inp = input("👤 请输入问题: ").strip() or "天气如何呢?"
        run_agent(agent, inp, config, ctx, args.output_mode)
        if args.show_tokens:
            print_token_usage(token_cb)


if __name__ == "__main__":
    main()
