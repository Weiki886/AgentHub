# AgentHub — AI 算法专家智能体团队

> **你的算法领域专家团队** — 从推荐系统到时序预测，从 CV 到 NLP，每个智能体都是一位专注于特定算法领域的高级专家，带来专业的方法论、工具链和落地经验。

---

![Stars](https://img.shields.io/github/stars/Weiki886/AgentHub?style=flat-square)
![License](https://img.shields.io/github/license/Weiki886/AgentHub?style=flat-square)
![Issues](https://img.shields.io/github/issues/Weiki886/AgentHub?style=flat-square)
![Forks](https://img.shields.io/github/forks/Weiki886/AgentHub?style=flat-square)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](https://github.com/Weiki886/AgentHub/pulls)

## 项目规模


| 层次           | 数量  | 描述                        |
| ------------ | --- | ------------------------- |
| 🤖 算法专家（技术层） | 80  | 12 个算法领域的工程师智能体，专注技术实现    |
| 📊 业务顾问（业务层） | 12  | 对应领域的业务分析师智能体，专注问题诊断与方案评估 |
| 🌏 算法领域      | 12  | 覆盖推荐、搜索、NLP、CV、风控、时序等核心方向 |


---

## 两层架构

AgentHub 采用**技术层 + 业务层**双层设计：

```
用户/产品经理
    ↓ 描述业务问题
业务层 — 业务分析师/顾问智能体
    ↓ 给出技术方向、评估 ROI
技术层 — 算法工程师智能体
    ↓ 实现方案、输出代码
```

- **业务层**：诊断业务问题 → 判断用什么算法方向 → 评估投入产出比（不写代码）
- **技术层**：理解业务需求 → 选择具体模型/算法 → 给出完整实现方案（写代码）

---

## 快速开始

### 一键安装到 AI 工具

支持 **8 种主流 AI 编程工具**，在 AgentHub 目录下执行：

```bash
# 自动检测已安装的工具，一键安装
./scripts/install.sh

# 指定安装到特定工具
./scripts/install.sh --tool cursor      # Cursor
./scripts/install.sh --tool windsurf     # Windsurf
./scripts/install.sh --tool antigravity  # Antigravity
./scripts/install.sh --tool gemini-cli   # Gemini CLI
./scripts/install.sh --tool opencode     # OpenCode
./scripts/install.sh --tool openclaw     # OpenClaw
./scripts/install.sh --tool aider        # Aider
./scripts/install.sh --tool qwen         # Qwen Code
```

> 部分工具需要先运行 `./scripts/convert.sh` 转换格式。

### 激活智能体

安装后在 AI 工具中直接引用智能体名称即可激活：

```
"使用推荐系统工程师帮我设计一个协同过滤推荐方案"
"用业务分析师帮我判断这个问题应该用时序预测还是异常检测"
```

---

## 算法领域

### 技术层 — 12 个算法领域


| 领域                            | 智能体数量 | 核心能力                                  |
| ----------------------------- | ----- | ------------------------------------- |
| **ab-testing**                | 6     | A/B 实验设计、分层实验、序贯检验、因果推断               |
| **anomaly-detection**         | 6     | 统计异常检测、时序异常、图异常、工业异常                  |
| **clustering-classification** | 7     | 聚类算法、分类模型、集成学习、模型可解释性                 |
| **cv**                        | 7     | 图像分类、目标检测、语义分割、OCR、视频理解、文生图           |
| **fraud-detection**           | 6     | 交易风控、信用风控、欺诈检测、账户安全                   |
| **graph**                     | 6     | 图神经网络、知识图谱、社区发现、链接预测、图数据库             |
| **nlp-text**                  | 8     | 文本分类、NER、信息抽取、文生文、语义匹配、情感分析           |
| **personalization**           | 6     | 推荐系统工程化、内容理解、用户画像、Push 策略             |
| **recommendation**            | 9     | 协同过滤、排序模型、多目标优化、冷启动、多样性优化             |
| **reinforcement-learning**    | 6     | 深度强化学习、离线 RL、Multi-Agent RL、RL 应用与安全  |
| **search-ranking**            | 8     | 语义搜索、Query 理解、Learning to Rank、知识图谱搜索 |
| **time-series**               | 6     | 时序预测、能源预测、金融时序、医疗时序、信号处理              |


### 业务层 — 12 个领域顾问


| 领域           | 智能体数量 | 核心能力                          |
| ------------ | ----- | ----------------------------- |
| **business** | 12    | 各算法领域的业务诊断、需求评估、ROI 分析、技术选型建议 |


---

## 支持的工具


| 工具              | 安装方式                                      | 格式说明                               |
| --------------- | ----------------------------------------- | ---------------------------------- |
| **Cursor**      | `./scripts/install.sh --tool cursor`      | `.cursor/rules/*.mdc` 项目级          |
| **Windsurf**    | `./scripts/install.sh --tool windsurf`    | `.windsurfrules` 项目级               |
| **Antigravity** | `./scripts/install.sh --tool antigravity` | `~/.gemini/antigravity/skills/` 全局 |
| **Gemini CLI**  | `./scripts/install.sh --tool gemini-cli`  | `~/.gemini/extensions/` 全局         |
| **OpenCode**    | `./scripts/install.sh --tool opencode`    | `.opencode/agents/` 项目级            |
| **OpenClaw**    | `./scripts/install.sh --tool openclaw`    | `~/.openclaw/` 全局                  |
| **Aider**       | `./scripts/install.sh --tool aider`       | `CONVENTIONS.md` 项目级               |
| **Qwen Code**   | `./scripts/install.sh --tool qwen`        | `.qwen/agents/` 项目级                |


---

## 脚本工具


| 脚本                         | 功能                                        |
| -------------------------- | ----------------------------------------- |
| `./scripts/convert.sh`     | 将 .md 转换为各工具专用格式（8 种），输出到 `integrations/` |
| `./scripts/install.sh`     | 将转换后的文件安装到对应工具的配置目录                       |
| `./scripts/lint-agents.sh` | 验证所有智能体文件的格式规范性                           |
| `./scripts/sync-tw.sh`     | 生成繁体中文版 README（需先创建 README.md）            |


### convert.sh 高级用法

```bash
# 仅转换技术层智能体
./scripts/convert.sh --domain technical

# 仅转换业务层智能体
./scripts/convert.sh --domain business

# 仅生成 Cursor 格式
./scripts/convert.sh --tool cursor

# 指定输出目录
./scripts/convert.sh --out /自定义路径/integrations
```

---

## 添加新智能体

1. 在对应领域目录下创建 `.md` 文件，命名规范：
  - 技术层：`nlp-text/sentiment-analysis-engineer.md`
  - 业务层：`business/nlp-business-advisor.md`
2. 文件必须包含 YAML frontmatter：
  ```yaml
   ---
   name: 智能体名称
   description: 一句话描述专业能力和擅长方向
   color: pink  # 颜色标识（业务层必须，技术层可选）
   ---
  ```
3. 运行格式验证：`./scripts/lint-agents.sh`
4. 运行转换生成：`./scripts/convert.sh`

---

## 与原项目的区别


|        | agency-agents（原项目） | AgentHub（本项目）            |
| ------ | ------------------ | ------------------------ |
| **定位** | 全领域通用智能体           | 算法领域垂直专家                 |
| **覆盖** | 17 个方向（前端/产品/运营等）  | 12 个算法领域（推荐/搜索/NLP/CV 等） |
| **架构** | 单层                 | 技术层 + 业务层双层协作            |
| **场景** | 产品开发全流程            | 算法研发全流程（诊断→设计→实现）        |


---

## 目录结构

```
AgentHub/
├── ab-testing/          # A/B 测试与实验设计
├── anomaly-detection/   # 异常检测
├── business/            # 业务分析师（业务层）
├── clustering-classification/  # 聚类与分类
├── cv/                 # 计算机视觉
├── fraud-detection/    # 欺诈检测
├── graph/              # 图计算与知识图谱
├── integrations/       # 各工具转换后的格式（由 convert.sh 生成）
├── nlp-text/           # 自然语言处理
├── personalization/     # 个性化推荐
├── recommendation/     # 推荐系统
├── reinforcement-learning/  # 强化学习
├── scripts/            # 工具脚本
├── search-ranking/      # 搜索与排序
└── time-series/        # 时序分析与预测
```

---

## 贡献指南

欢迎参与贡献！不管是新增算法领域智能体、完善现有智能体的内容，还是改进文档，都非常欢迎。

### 贡献方式

#### 1. 新增智能体

在对应目录下创建 `.md` 文件，命名规范：

```bash
# 技术层：<领域>/<功能>-engineer.md
nlp-text/sentiment-analysis-engineer.md

# 业务层：business/<功能>-business-advisor.md
business/nlp-business-advisor.md
```

文件格式要求：

```markdown
---
name: 智能体名称
description: 一句话描述专业能力和擅长方向
color: pink  # 颜色标识（业务层必须，技术层可选）
---

# 智能体名称

你是**智能体名称**，[一句话定位]。

## 你的身份与记忆
- **角色**：具体角色
- **个性**：性格特点
- **记忆**：记住什么
- **经验**：擅长什么

## 核心使命
具体职责和工作内容。

## 关键规则
做事的原则和红线。

## 技术交付物
代码示例、模板、框架等具体产出。

## 工作流程
分步骤的工作流。

## 沟通风格
说话的方式和语气示例。
```

#### 2. 完善现有智能体

发现内容不完整、示例代码过时或表达不够准确，随时改进。

#### 3. 改进文档

发现文档有误或可以写得更好，欢迎提交。

### 提交规范

1. Fork 本仓库
2. 创建分支：`git checkout -b add-xxx-engineer`
3. 新增或修改内容后，运行格式验证：
  ```bash
   ./scripts/lint-agents.sh
  ```
4. 如有智能体变更，重新生成集成文件：
  ```bash
   ./scripts/convert.sh
  ```
5. 提交 PR，简单说明做了什么

### 约定

- 文件使用 LF 换行符（不要 CRLF）
- 一个 PR 做一件事，不要把新增智能体和文档修改混在一起
- commit message 使用中文

---

## 许可证

本项目基于 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) 开源。

你可以自由地：

- 自由使用、复制、修改
- 分发软件副本
- 私人或商业用途

前提条件：

- 必须保留许可证和版权声明
- 如有修改，必须明确说明
- 不得使用 "AgentHub" 名称或项目贡献者的名义进行推广

---

## 致谢

### 灵感来源

本项目的设计思路和架构参考自 [jnMetaCode/agency-agents-zh](https://github.com/jnMetaCode/agency-agents-zh)，该项目提供了将 AI 智能体系统化管理并支持多种 AI 编程工具的完整方案，是本项目的重要参考。

### 上游项目

本项目的设计基于上游英文版项目 [msitarzewski/agency-agents](https://github.com/msitarzewski/agency-agents)，感谢原作者创建了这个出色的多智能体人设框架。