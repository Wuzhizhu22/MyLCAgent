# LangChain Based Agent

个人学习目的使用 LangChain 框架构建的简易智能体，支持工具调用和结构化响应。

## 功能特性

- 基于 LangChain 框架构建的智能体
- 支持工具调用
- 结构化响应输出
- 使用 YAML 文件进行配置管理
- 支持对话上下文管理

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

修改 `tools.yaml` 文件配置 API 密钥和模型参数：

```yaml
llm:
  model: gpt-4.1
  api_key: YOUR_API_KEY
  base_url: YOUR_BASE_URL
```

## 使用示例

运行智能体：

```bash
uv run main.py
```

## 项目结构

```
langchain-based-agent/
├── main.py          # 主程序
├── tools.yaml       # 配置文件
├── pyproject.toml   # 项目依赖
├── README.md        # 项目说明
└── .gitignore       # Git忽略文件
```
