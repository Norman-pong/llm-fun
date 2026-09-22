# DESIGN.md — llm-fun 设计文档

> 本文档是仓库的设计依据与事实来源，供后续会话（含 AI agent）迭代时遵循。
> 结构性决策变更必须先改本文档、再动代码，并通过 lore commit 留痕。
> 三文档分工：**DESIGN.md=设计依据**（稳定，低频改）；**README.md=路线图与进度**（唯一进度源）；**AGENTS.md=agent 行为规则**（工具链命令与 lore 协议）。

---

## 1. 项目定位与成功标准

个人学习大模型理论与工程的参考实现仓库：**理论线从零实现**（autograd → 语言模型 → GPT → 预训练），**实践线做应用工程**（API → RAG）。

**学习模式（2026-09-22 起）**：仓库由 agent 完整迭代为「参考答案库 + 验收测试库」，每章完成时打 tag（`t1`…`p2`），学习者通过 tag 定位各阶段快照，按自身节奏学习；个人笔记在学习者自己的外部项目中维护，不写入本仓库。

**总成功标准（可验证）**：7 章（T1-T5、P1-P2）全部实现完成、各自 DoD 测试全绿、每章有 tag；P2 的 RAG demo 本地可跑（生成环节除外，见 3.1 P2 备注）。

时间假设：业余 6h/周，7 章学习约 20 周（学习者节奏）；实现由 agent 一次性完成。

## 2. 需求与决策记录

已确认的需求（2026-09-22）：

| 需求项 | 决定 |
|---|---|
| 学习侧重 | 理论+实践双线 |
| 技术栈 | Python 为主（PyTorch） |
| 学习者基础 | 资深工程师、ML 入门 |
| 产出形态 | 笔记 + 可运行代码 |

两次方案评审（冷眼评审）后的关键决策，细节见 git log 中的 lore trailer：

| 决策 | 依据 |
|---|---|
| 承诺范围砍到 T1-T5+P1-P2，其余进 backlog | 18 仓库三线并行是烂尾头号风险；backlog 写了不算欠 |
| 每章必须有 DoD + 时间盒 + 超时降级路径 | 「先定可验证成功标准」；业余学习无止损必弃 |
| AGENTS.md 去前端化（Vite+ → uv） | Python 为主，vp 规则造成持续摩擦 |
| 本地进阶只用 Apple Silicon 原生工具（MLX/llama.cpp） | LlamaFactory/vLLM 是 CUDA 生态，Mac 不可运行 |
| 进度单一来源：仅根 README checklist | 三层进度追踪违反「单一数据源」 |
| 测试进工作流（pytest 验 DoD） | 「做完：跑最小验证，不只编译过」 |
| 每章 1 主线 + ≤1 辅助参考 | 多参考源 = FOMO 囤资源，读不完 |
| 学习模式改为 agent 完成制 + tag 导航 + 笔记外置 | 学习者按 tag 定位阶段快照自学，仓库保持纯参考实现；笔记在个人项目维护，避免双重维护负担 |

## 3. 课程设计

### 3.1 承诺范围（7 章，每章 DoD 用 pytest 验证）

理论线 `theory/`：

| 章 | 时间盒 | 主参考 | DoD | 超时降级 |
|---|---|---|---|---|
| T1 autograd 与反向传播 | 2 周 | [micrograd](https://github.com/karpathy/micrograd) + [zero-to-hero 视频 L1](https://karpathy.ai/zero-to-hero.html) | 手写 autograd 与 torch.autograd 在 ≥10 个随机复合函数上梯度对齐（float32, atol=1e-5）；训小 MLP loss 单调下降 | 精读源码+笔记，实现仅到标量加乘 |
| T2 语言建模直觉 | 3 周 | [makemore](https://github.com/karpathy/makemore)（bigram→MLP 两讲） | 人名数据集 MLP loss 低于 bigram 基线；采样出可读人名；笔记含激活/梯度诊断 | 砍激活诊断，保 bigram+MLP |
| T3 分词器 | 2 周 | [minBPE](https://github.com/karpathy/minBPE) + [tokenizer 视频](https://www.youtube.com/watch?v=zduSFxRajkE) | BPE 编码→解码 roundtrip 测试通过；与 minbpe basic 在样例语料上编码结果一致 | 只实现 basic BPE，不做 GPT-2 正则版 |
| T4 Attention 与 GPT | 4 周 | [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) ch3-4；辅 [build-nanogpt](https://github.com/karpathy/build-nanogpt) 讲座 | 手写 multi-head attention 与 PyTorch 官方实现数值对齐；GPT 前向 shape/数值检查通过；能采样生成文本 | 超 6 周：精读 build-nanogpt + 只实现 attention 模块 |
| T5 预训练 | 3 周 | LLMs-from-scratch ch5 + 附录 D（LR 预热/余弦衰减/梯度裁剪） | 莎士比亚语料 MPS 训练 loss 持续下降、生成可读风格文本；记录训练曲线与超参 | 缩语料/步数，保「训练闭环跑通」 |

实践线 `practice/`：

| 章 | 时间盒 | 主参考 | DoD | 超时降级 |
|---|---|---|---|---|
| P1 API 与 Prompt | 2 周 | [llm-cookbook](https://github.com/datawhalechina/llm-cookbook) 必修前两门（ChatGPT Use / API） | 封装 OpenAI 兼容客户端（.env 加载/重试/结构化输出），**mock 测试全绿**；prompt 对照实验脚本一键生成结果表（真实运行依赖 API key，见备注） | 砍对照组数量，保客户端封装 |
| P2 RAG | 4 周 | [rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)（Part 1-9 基础部分） | 端到端检索问答可跑（Chroma 嵌入式 + 本地 TF-IDF embedding，**零 API key、零服务依赖**）；≥20 条评测集的检索 hit-rate 有数字；回答生成环节走 P1 客户端，无 key 时跳过（见备注） | 砍 rerank/混合检索，保检索+生成闭环 |

> **P1/P2 API 依赖备注（2026-09-22）**：与「密钥只走环境变量」约束一致，agent 不持有 API key。P1 的 prompt 实验与 P2 的回答生成需学习者配好 `.env` 后运行对应脚本（脚本会输出结果表/答案）；两者的代码正确性均由无 key 的 mock/本地测试验证。

论文精读只承诺 3 篇强绑定主线：BPE（T3）、Attention is All You Need（T4）、GPT-2（T4/5）。笔记放 `papers/`。

### 3.2 Backlog（触发条件到了才升级，写了不算欠）

| 项 | 触发条件 | 要点 |
|---|---|---|
| T6 现代架构 | T5 完成且想深入 | LLMs-from-scratch bonus（Llama 化/GQA/RoPE），MoE 可选 |
| T7 微调与对齐 | T6 后或需要时 | LLMs-from-scratch ch6-7 + 附录 E（LoRA）；reasoning-from-scratch 选读 |
| T8 毕业项目 | T7 后 | 想自己训→[minimind](https://github.com/jingyaogong/minimind)（云租 3090，¥10 内跑 26M 全流程）；只收口认知→[nanochat](https://github.com/karpathy/nanochat) 精读；超时降级→只精读 tokenizer/训练循环/SFT 三模块 |
| P3 Agent | P2 后 | [agents-course](https://github.com/huggingface/agents-course) Unit1-3（有中文版）+ [MCP quickstart](https://modelcontextprotocol.io/docs/getting-started/intro) |
| P4 微调实践 | T7 后 | 本地主线 MLX（mlx-lm LoRA）；LlamaFactory 为云上环节 |
| P5 部署与评测 | P3/P4 后 | llama.cpp 本地服务 + promptfoo；vLLM/lm-eval-harness 云上 |

### 3.3 止损规则

连续两章超时 50% → 暂停开新章，只收尾已开章节；每月末做一次进度 review。

## 4. 章节协议

- **目录命名**：`theory/NN-topic/`、`practice/NN-topic/`，零填充编号；`papers/` 一篇一文件。
- **每章结构**：`README.md`（章模板）+ `notes.md`（agent 写的讲解笔记：推导/实验数据/踩坑，供学习者阅读）+ `code/*.py`（定稿参考实现）+ `test_*.py`（DoD 验收测试，也是学习者自己重写时的对拍标准）+ 小数据文件。
- **tag 制**：每章完成时打 tag（`t1`…`p2`，附中文说明），学习者 `git checkout <tag>` 或直接浏览对应章节目录定位学习快照。
- **章 README 模板（三段）**：① 学习目标（含 DoD 原文与达成情况）② 资源（1 主线 + ≤1 辅助，带一句「为什么读」）③ 复盘（agent 的实现复盘：关键取舍、实验数据）。
- **代码形态**：`.py` + `notes.md` 为定稿（可 ruff、可 pytest）；notebook 仅探索用、提交前清空输出、不作为定稿。
- **数据规则**：小数据文件直接进仓库（`names.txt`、莎士比亚语料等）；大数据进 `data/`（已 ignore）但章 README 记来源 + 下载脚本。
- **共享代码**：第二处重复时抽公共函数；模块化抽象等第三处重复再说，暂不建 `src/` 包。

## 5. 工具链约定

- **uv**：`uv sync --frozen` 装环境；`uv add` 加依赖；运行一律 `uv run`。`uv.lock` 与 `.python-version` 必须提交。
- **质量**：提交前 `uv run ruff check .` 通过；DoD 用 `uv run pytest` 验证。
- **密钥**：API key 只走 `.env`（python-dotenv 加载），`.env` 不进 git；模板见 `.env.example`。
- **notebook**：VS Code + ipykernel（已装），不装 jupyterlab。

## 6. 算力策略（本机无 NVIDIA GPU）

原则：**承诺范围零算力成本，进阶本地走 Apple Silicon 原生路径，训练按预算租云。**

### 承诺 7 章全部本机可跑

| 章 | 算力需求 | 跑法 |
|---|---|---|
| T1 | 标量运算，CPU 秒级 | CPU |
| T2 | 万级参数 MLP，分钟级 | CPU/MPS |
| T3 | 纯字符串算法 | CPU |
| T4 | ~10M 参数 mini-GPT，前向/采样秒~分钟 | MPS |
| T5 | 莎士比亚语料 ~1MB，训练几十分钟级 | MPS，降级=缩语料/步数 |
| P1 | 零本地算力 | 调 API |
| P2 | Chroma 嵌入式 + API embedding | CPU |

### Backlog 算力出口

- 本地微调：**MLX（mlx-lm LoRA）**，Apple 官方框架，统一内存架构原生优化。
- 本地推理：**llama.cpp**（GGUF 量化，16GB 内存 Mac 可跑 7B~14B）。
- CUDA-only 工具（LlamaFactory/vLLM/lm-eval-harness）一律标注「云上环节」，不在本机硬跑。
- 云租预算封顶：AutoDL 3090 ~¥2/hr、4090 ~¥4/hr；**单章 ≤¥50，累计 ≤¥200**，超限缩模型规模不硬扛。

### MPS 已知坑（必读）

- 不支持 float64，全程 float32；数值比对用宽松 atol（1e-5 而非 1e-6）。
- 缺算子时设 `PYTORCH_ENABLE_MPS_FALLBACK=1`（会**静默**回 CPU 变慢，先确认是否需要）。
- 长训练警惕 Metal watchdog 与统一内存竞争；T5 只跑小语料短程。

## 7. 参考项目全景（2026-09-22 调研沉淀，star 为当日实测）

> 用途：防止重复调研与重复决策。已被主线引用的项目见第 3 节；下表补充定位与备注。

### 理论线

| 项目 | star | 定位 / 在本仓库何处被引用 | 备注 |
|---|---|---|---|
| [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | 105k | T4/T5/T6/T7 主线教材 | 每章 `01_main-chapter-code` + 编号 bonus + 习题解答；本仓库章节协议的头号模板 |
| [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) | 24.5k | T1-T4 配套视频课总入口 | 已停更但内容定稿；螺旋式课程弧线 |
| [karpathy/micrograd](https://github.com/karpathy/micrograd) | 17.6k | T1 主线 | ~100 行 autograd |
| [karpathy/makemore](https://github.com/karpathy/makemore) | 4.3k | T2 主线 | notebook 即讲义 |
| [karpathy/minBPE](https://github.com/karpathy/minbpe) | 10.7k | T3 主线 | basic/regex/gpt4 分层递进 |
| [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) | 5.5k | T4 辅助 | 3.5h 逐行手写 GPT-2 |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 63k | T4/T5 参考 | 可复现实验脚手架的范本 |
| [karpathy/nanochat](https://github.com/karpathy/nanochat) | 58k | backlog T8 候选 | 2025 旗舰：tokenizer→预训练→SFT→RL→推理全生命周期 |
| [jingyaogong/minimind](https://github.com/jingyaogong/minimind) | 62k | backlog T8 首选 | 中文、单卡训 26M，`train_*.py` 脚本即章节 |
| [datawhalechina/llms-from-scratch-cn](https://github.com/datawhalechina/llms-from-scratch-cn) | 4.4k | 中文对照（按需） | **以英文原版代码为准**，翻译滞后 |
| [rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) | 5.3k | backlog T7 选读 | GRPO/RLVR 从零实现 |
| [Stanford CS336](https://cs336.stanford.edu/) | — | backlog 参考 | 作业设计金标准（A1-A5 对应生命周期五环节） |
| [labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) | 67.5k | 论文精读代码对照 | 60+ 论文旁注实现 |
| [d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh) | 81k | DL 底座补漏（按需） | 前置基础缺什么补什么 |

### 实践线

| 项目 | star | 定位 / 引用处 | 备注 |
|---|---|---|---|
| [datawhalechina/llm-cookbook](https://github.com/datawhalechina/llm-cookbook) | 25k | P1 主线 | 必修/选修分级；2025-06 后停更但入门内容稳定 |
| [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch) | 9.4k | P2 主线 | 官方 18 部分渐进 |
| [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) | 30k | P2 进阶卡片（按需） | 42+ 技巧一 notebook 一技巧 |
| [datawhalechina/llm-universe](https://github.com/datawhalechina/llm-universe) | 14k | P2 中文入门（按需） | 项目式：个人知识库助手 |
| [huggingface/agents-course](https://github.com/huggingface/agents-course) | 33k | backlog P3 主线 | 有中文版；课程+GAIA 基准+认证闭环 |
| [microsoft/ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) | 75k | backlog P3 知识清单 | 18 课大纲；代码样例绑定 Azure 需改造 |
| [NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents) | 24k | backlog P3 实验骨架 | 50+ 渐进式 agent 模式 notebook |
| [huggingface/smol-course](https://github.com/huggingface/smol-course) | 6.8k | backlog P4 原理层 | 小模型后训练实战 |
| [hiyouga/LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) | 75k | backlog P4 云上工具 | **CUDA 生态，Mac 不可运行** |
| [datawhalechina/self-llm](https://github.com/datawhalechina/self-llm) | 32k | backlog P4 中文参考 | 含 Apple Silicon/MLX 章节 |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | 129k | backlog P5 本地推理 | Apple Silicon 端侧首选 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 92k | backlog P5 云上服务 | **Mac 不可本地运行** |
| [promptfoo/promptfoo](https://github.com/promptfoo/promptfoo) | 25k | backlog P5 应用层评测 | 对接 llama.cpp 的 OpenAI 兼容接口可本地闭环 |
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 14k | backlog P5 模型层评测 | 不直接评测 GGUF |

中文资源总体结论：入门段（API/Prompt/RAG）中文成体系；**Agent 段中文明显滞后，以英文官方课为主**。

## 8. 后续 agent 迭代指引

**学习模式（2026-09-22 起）：agent 完成制 + tag 导航。**

**agent 完成一章的标准流程：**

1. 读 `DESIGN.md` 第 3 节找到该章的 DoD、参考与降级路径。
2. 改前查询 lore：`lore constraints/rejected <章节路径> --json`。
3. 按章节协议（第 4 节）建目录，写章 README（目标+资源）、notes.md（讲解笔记）、参考实现与 DoD 测试。
4. **pytest 达成 DoD** + `uv run ruff check .` 通过，缺一不可。
5. `lore commit` 中文提交，然后打 tag：`git tag -a <tN|pN> -m "<章名>：DoD 达成"`，并更新根 README 状态表。

**学习者的使用方式（写在根 README，agent 保持其有效）：**

- `git tag` 看全部章节快照；`git checkout tN -- theory/`（或直接浏览章节目录）定位某一章。
- 推荐学法：先看该章 README 资源 → 自己动手写 → 用章内 `test_*.py` 验收自己的实现 → 对照参考实现查漏。
- 个人笔记在学习者自己的外部项目维护，不写入本仓库。

**改设计先改文档：** 任何结构性变更（章节增删、DoD 修改、工具链更换、预算调整）必须先更新 DESIGN.md 对应小节，并在 commit body 里说明原因。

**本文档修改规则：** 第 3-6 节的变更必须走 lore commit 留痕；错别字与链接修复无需。

**硬件红线：** 本地不引入 CUDA-only 工具；云租超预算（第 6 节）先提案再执行。
