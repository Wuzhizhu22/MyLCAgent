# Changelog

## [0.1.0] - 2025-12-30

### Added

- 初始项目搭建，基于 LangChain 框架的智能体实现
- 支持工具调用功能
- 结构化响应输出
- 对话上下文管理
- 能够实现对话
- 动态模型选择功能(根据对话长度自动切换模型)

### Fixed

- 优化了输出格式，使响应更易读

## [0.1.1] - 2025-12-31

### Added

- 新增了流式传输的样例

## [0.1.2] - 2025-12-31

### Added

- 新增了可选择运行模式（stream/invoke）的样例

### Fixed

- 把 tools.yaml 重命名为 llm.yaml，用于配置 llm 模型参数

## [0.1.3] - 2025-12-31

### Fixed

- 结构化消息更加清晰且具有更高的扩展性，方便未来添加更多功能

## [0.1.4] - 2025-12-31

### Fixed

- 仅在 @tool 装饰器中添加明确的 description 参数，提升 LLM 对工具的理解准确性
- 引入 Pydantic 模型，细化 tool 输入
- 移除不必要的 uuid 导入和消息 ID 生成逻辑，简化代码结构

## [0.2.0] - 2025-12-31

### Added

- 实现完整的 ReAct 格式输出（Question, Thought, Action, Observation, Final Answer）
- 新增统一的 ReAct 步骤打印函数 `print_react_step`，支持所有 ReAct 字段的格式化输出
- 新增消息解析分发函数 `dispatch_react_elements`，自动识别并打印不同类型的消息内容
- 添加对话模式支持，通过 `--conversation` 参数实现连续对话交互
- 新增 Token 使用统计功能，通过 `--show-tokens` 参数控制是否显示
- 引入 SummarizationMiddleware 中间件，自动进行消息摘要以优化上下文管理

### Fixed

- 重构 `run_agent` 函数，统一处理 stream 和 invoke 两种运行模式，减少代码重复
- 优化 ResponseFormat 数据结构，包含完整的 ReAct 字段（question, thought, action, observation, final_answer）
- 优化 SYSTEM_MESSAGE 提示词，明确要求使用 ReAct 格式输出
- 简化输出处理逻辑，通过统一的辅助函数提升代码可读性和可维护性
