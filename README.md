# LangChain Based Agent

个人学习目的使用 LangChain 框架构建的简易智能体，支持工具调用和结构化响应。

## 功能特性

- 基于 LangChain 框架构建的智能体
- 支持工具调用
- 结构化响应输出
- 使用 YAML 文件进行配置管理
- 支持对话上下文管理
- **ReAct 输出格式**：清晰的思考过程展示（💭 Thought、🔧 Action、🔍 Observation、✅ Final Answer）

## 安装说明

1. 克隆仓库：

   ```bash
   git clone https://github.com/Wuzhizhu22/MyLCAgent.git
   cd MyLCAgent
   ```

2. 安装依赖：
   ```bash
   uv sync
   ```

## 配置说明

修改 `llm.yaml` 文件配置 API 密钥和模型参数：

```yaml
llm:
  model: gpt-4.1
  api_key: YOUR_API_KEY
  base_url: YOUR_BASE_URL
```

## 使用示例

运行智能体：

```bash
# 单次对话模式（默认），使用 invoke 输出
uv run main.py

# 单次对话模式，使用 stream 流式输出
uv run main.py --output-mode stream

# 多轮对话模式，使用 invoke 输出
uv run main.py --conversation

# 多轮对话模式，使用 stream 流式输出
uv run main.py --conversation --output-mode stream

# 显示 Token 使用统计
uv run main.py --show-tokens
```

### 参数说明

- `--conversation`: 启用多轮对话模式（默认为单次对话）
- `--output-mode`: 输出方式，可选值为 `stream`（流式输出）或 `invoke`（一次性输出，默认）
- `--show-tokens`: 显示 Token 使用统计（默认不显示）

### ReAct 输出格式说明

智能体使用 ReAct（Reasoning + Acting）模式进行推理和行动，输出格式如下：

#### 模式输出示例

'''
👤 请输入问题: 天气如何呢

🤔 智能体正在思考 (invoke)...

🔧 Action: get_user_location

🔍 Observation: 佛罗里达

🔧 Action: get_weather_for_location(query={'city': '佛罗里达'})

🔍 Observation: 佛罗里达总是阳光明媚，真是‘晴’深似海！

❓ Question: 天气如何呢

💭 Thought: 用户询问天气情况，但没有指定具体位置。我需要先获取用户当前位置，然后查询该位置的天气信息。

✅ Final Answer: 佛罗里达总是阳光明媚，真是'晴'深似海！这里的天气就像佛罗里达的橙子一样——鲜亮又充满活力！☀️
'''

#### 输出符号说明

- 💭 **Thought**: 智能体的思考过程（推理步骤）
- 🔧 **Action**: 智能体执行的工具调用
- 🔍 **Observation**: 工具执行的结果
- ✅ **Final Answer**: 智能体的最终答案
- 🎯 **自定义输出**: 工具内部的调试信息

## 项目结构

```
langchain-based-agent/
├── main.py          # 主程序
├── tools.yaml       # 配置文件
├── pyproject.toml   # 项目依赖
├── README.md        # 项目说明
└── .gitignore       # Git忽略文件
```
