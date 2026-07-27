# 需求文档：aiida-uranium-workflow

> 状态：示例（Draft v0.1）
> 作者：aiida-uranium-workflow 团队
> 最后更新：2026-07-27
> 关联代码：`src/aiida_uranium_workflow/`

## 1. 背景

`aiida-uranium-workflow` 是一个基于 [AiiDA](https://www.aiida.net/) 的第一性原理高通量计算工作流库，专注于 **铀（U）** 体系的 DFT 计算。铀是强关联金属，其 5f 电子具有局域化与巡游性共存的特点，常规 VASP/ABACUS 单点计算需要配合：

- **k 点 / 截断能收敛性测试（convergence）**：找到精度-成本平衡点
- **展宽（smear）扫描**：解决费米面附近电子占据不稳定的问题
- **磁矩（magmom）初始化搜索**：避免磁性体系陷入高能亚稳态

历史痛点：每个工作流都是独立脚本，CLI 入口分散（`convergence_run.py`、`smear_run.py` 等），用户跨工作流时需要重复加载结构、赝势族、计算资源。

## 2. 目标

提供一个 **统一、声明式、可复现** 的工作流框架：

| 目标 | 衡量标准 |
|---|---|
| 统一 CLI 入口 | 单一 `aiida-uranium-workflow` 命令覆盖 convergence / smear / magmom |
| 配置与代码解耦 | 所有物理参数由 `parameters/<code>/*.yml` 驱动，不需改代码 |
| 多 code 适配 | 同一份 JSON 输入可分别跑 ABACUS 和 VASP，结果可对比 |
| 多结构批处理 | 单次提交可对若干 `StructureData` 启动多组 workchain |
| 可复现 | 完整 metadata（结构、赝势族、机器、code、label）落入 AiiDA provenance graph |

## 3. 范围

### 3.1 In Scope

- 三类工作流：convergence、smear、magmom
- 两种 DFT code：ABACUS、VASP
- 统一 CLI（`src/aiida_uranium_workflow/cli/main.py`）
- 输入构造器（`input_builders/`）：基于 JSON + YAML 参数生成 AiiDA inputs
- 调度器（`schedulers/`）：将多组参数空间映射为 workchain
- 报告生成（`utils/report/`）：CSV / JSON / Markdown 表格
- AE-EOS 参考数据集（`static/AE_EOS/`）用于基准对照

### 3.2 Out of Scope

- 自定义 AiiDA plugin 注册（沿用 `aiida-abacus`、`aiida-vasp`）
- 其他 DFT code（Quantum ESPRESSO、CP2K 等）
- 强关联求解器（DFT+U、DMFT）参数扫描
- Web UI / 任务看板

## 4. 用户故事

| ID | 角色 | 故事 | 优先级 |
|---|---|---|---|
| US-1 | 计算科学家 | "我准备一个 JSON 输入，希望用一行命令分别在 ABACUS 和 VASP 上跑 k 点收敛测试。" | P0 |
| US-2 | 课题组新人 | "我不懂 AiiDA，只想用 `aiida-uranium-workflow smear --input foo.json` 启动计算。" | P0 |
| US-3 | 数据分析 | "计算完成后，我想直接拿到一份 CSV / Markdown 表格，包含能量、磁矩、收敛状态。" | P1 |
| US-4 | 高通量管理员 | "我想对 10 个不同结构同时跑 magmom 搜索，避免重复输入。" | P1 |
| US-5 | 复现者 | "我拿到别人共享的 `inputs.json` + commit hash，能完全复现他的工作流。" | P2 |

## 5. 功能需求

### 5.1 统一 CLI（对应 US-1 / US-2）

```
aiida-uranium-workflow <subcommand> [options]
  convergence     k 点 / 截断能收敛性扫描
  smear           展宽参数扫描
  magmom          磁矩初始化搜索
  structure       结构查看 / 校验
```

- 支持 `--code <abacus|vasp>` 切换 DFT 后端
- 支持 `--inputs <path.json>` 单一输入或 `--multi-inputs <dir>/` 多结构批处理
- 输出 `WorkChainNode` PK + 状态轮询提示

### 5.2 输入格式（JSON）

```json
{
  "structure_label": "bcc_U",
  "structure_file": "structures/bcc_U.cif",
  "pseudo_family": "sg15_sz",
  "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 24}
}
```

### 5.3 参数分层（YAML）

```
parameters/
  abacus/
    abacus.yml        # 全局 ABACUS 默认
    convergence.yml   # convergence 子工作流覆盖
    smear.yml
    magmom.yml
  vasp/
    vasp.yml
    convergence.yml
    smear.yml
    magmom.yml
```

### 5.4 报告生成（对应 US-3）

- 输出格式：CSV、JSON、Markdown
- 字段：`label`、`energy_per_atom`、`magnetization`、`converged`、`wall_time`
- 支持 VASP / ABACUS label 格式互转

### 5.5 工具函数

- `utils/cal_json.py`：从 AiiDA Calculation node 抽取关键结果
- `utils/copy_remote.py`：跨机器 workchain 节点复制
- `utils/labels.py`：label 解析与归一化
- `utils/structure.py`：结构预处理（去空、归一化、对称性探测）

## 6. 非功能需求

| 维度 | 要求 |
|---|---|
| Python | ≥ 3.9 |
| AiiDA | ≥ 2.0 |
| 测试 | `pytest` 覆盖 input_builder / scheduler / CLI / utils，CI 全绿 |
| 代码质量 | pre-commit (black, ruff, mypy) |
| 可移植性 | 单机、SLURM、SSH 三种 scheduler 都能工作 |
| 文档 | 本文件 + `README.md` + 各模块 docstring |

## 7. 架构概览

```
JSON inputs ──┐
YAML params ──┼─► InputBuilder ──► AiiDA WorkChain ──► DB
              │                            │
              │                            ▼
              └─► Scheduler ────────► Report (CSV/MD)
```

| 组件 | 路径 | 职责 |
|---|---|---|
| CLI | `cli/main.py` | 解析子命令、加载 JSON/YAML |
| InputBuilder | `input_builders/` | 生成 AiiDA inputs 字典 |
| Scheduler | `schedulers/` | 多组参数循环 → 提交 WorkChain |
| Workflow | `workflows/` | `ConvergenceWorkChain` / `SmearWorkChain` / `MagmomWorkChain` |
| Report | `utils/report/` | 结果汇总 |
| Config | `parameters/` | YAML 物理参数默认值 |

## 8. 验收标准

- [ ] 三类工作流（convergence / smear / magmom） × 两种 code（abacus / vasp）共 6 条路径全部跑通端到端（mock 或真实计算）
- [ ] CLI 单一入口 `aiida-uranium-workflow` 可发现全部子命令
- [ ] 多结构输入目录批处理可一次提交 ≥ 10 个 workchain
- [ ] 测试覆盖率 ≥ 80%
- [ ] `pytest tests/` 在干净环境下全绿
- [ ] `git clone` + `pip install -e .` 即可在 AiiDA 配置好的机器上直接运行示例

## 9. 风险与开放问题

| 风险 | 缓解 |
|---|---|
| ABACUS / VASP plugin 版本差异导致 input 字段不兼容 | `parameters/<code>/*.yml` 按 code 隔离 |
| 用户已有旧 CLI 脚本（`convergence_run.py` 等） | 文档迁移指南，旧脚本保留若干版本 |
| AiiDA provenance graph 膨胀 | workchain label 规范化（`utils/labels.py`） |
| 5f 强关联体系 SCF 不收敛 | magmom 工作流提供初始磁矩候选集 |

## 10. 里程碑

| 版本 | 范围 | 目标日期 |
|---|---|---|
| v0.1 (当前) | convergence + smear，ABACUS | 2026-Q3 |
| v0.2 | VASP 支持 | 2026-Q4 |
| v0.3 | magmom 工作流 | 2027-Q1 |
| v1.0 | 6 条路径全稳定，文档完善 | 2027-Q2 |

## 11. 附录

- A. 示例输入：`src/aiida_uranium_workflow/example/inputs.json`
- B. AE-EOS 基准：`src/aiida_uranium_workflow/static/AE_EOS/`
- C. 相关项目：`aiida-abacus`、`aiida-vasp`、`aiida-core`
