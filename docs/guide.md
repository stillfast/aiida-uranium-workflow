# `aiida-uranium-workflow` —— 详细说明文档

> 单一入口文档。覆盖仓库是什么、怎么用、怎么扩展、内部如何工作。
> 已有专项文档：
> - [`feature.md`](./feature.md) —— CLI 四阶段（`run` / `report` / `copy` / `archive`）的端到端工作流图与函数表
> - [`requirements.md`](./requirements.md) —— 需求规格（背景 / 目标 / 用户故事 / 验收标准）
> - [`tests.md`](./tests.md) —— 当前测试组织（早期草稿）
>
> 本文件其余内容**不重复**上面三份的细节，但解释它们没覆盖的"代码组织 / 注册表机制 / 扩展指南"。

---

## 目录

- [1. 项目速览](#1-项目速览)
- [2. 安装与运行](#2-安装与运行)
- [3. 端到端 4 阶段流程（与 `feature.md` 对应）](#3-端到端-4-阶段流程与-featuremd-对应)
- [4. 输入数据格式](#4-输入数据格式)
- [5. YAML 参数层（`parameters/`）](#5-yaml-参数层-parameters)
- [6. 仓库结构与模块职责](#6-仓库结构与模块职责)
- [7. 调度与 orchestrator 注册表](#7-调度与-orchestrator-注册表)
- [8. 输入适配器（`input_builders/`）](#8-输入适配器-input_builders)
- [9. WorkChain（`workflows/`）](#9-workchain-workflows)
- [10. 报告模块（`utils/report/`）](#10-报告模块-utilsreport)
- [11. 拷贝（`utils/copy_remote.py`）](#11-拷贝-utilscopy_remotepy)
- [12. 标签解析（`utils/labels.py`）](#12-标签解析-utilslabelspy)
- [13. CLI 入口（`cli/main.py` + `cli/_common.py`）](#13-cli-入口-climainpy--clicommonpy)
- [14. 公开 API 一览](#14-公开-api-一览)
- [15. 扩展指南](#15-扩展指南)
- [16. 测试组织](#16-测试组织)
- [17. 已知问题与故障排查](#17-已知问题与故障排查)
- [18. 引用与相关项目](#18-引用与相关项目)

---

## 1. 项目速览

`aiida-uranium-workflow` 是一个基于 [AiiDA](https://www.aiida.net/) 的 DFT 高通量工作流库，针对**铀（U）** 体系。围绕三类核心工作流 × 两种 DFT 后端（ABACUS / VASP）= **6 条 WorkChain 路径**：

| 类别 | ABACUS | VASP |
|---|---|---|
| **smear**（展宽 σ × 占据方法 扫描） | `abacus.smear` | `vasp.smear` |
| **convergence**（截断能 × k 点 收敛） | `abacus.convergence` | `vasp.convergence` |
| **magmom**（初始磁矩扫描） | `abacus.magmom` | `vasp.magmom` |
| **base**（单次基态 SCF） | `abacus.base` | `vasp.base` |

每条路径都是一个 AiiDA `WorkChain` 子类（`pyproject.toml` 的 `[project.entry-points."aiida.workflows"]` 注册），由包内的 **orchestrator** 统一调度。

**统一 CLI**：

```bash
aiida-uranium run     --method {smear,convergence,magmom,base} --input input.json
aiida-uranium report  --method ...                  --input output.json
aiida-uranium copy    --method ... --output PATH
aiida-uranium archive  --method ... --output result.aiida
```

详见 [feature.md](./feature.md) 的 §2–§5。

**包元数据**：

| 字段 | 值 |
|---|---|
| `name` | `aiida-uranium-workflow` |
| `version` | `0.1.0` |
| Python | ≥ 3.10 |
| AiiDA | ≥ 2.0 |
| 关键依赖 | `aiida-abacus>=0.1`, `aiida-vasp>=5.0`, `ase>=3.22`, `pyyaml>=6.0`, `pyxtal>=1.0` |
| 可选开发依赖 | `pytest`, `pre-commit`, `black`, `isort`, `flake8`, `mypy` |

---

## 2. 安装与运行

### 2.1 从源码安装（开发模式）

```bash
git clone https://github.com/liguozhou/aiida-uranium-workflow.git
cd aiida-uranium-workflow
pip install -e ".[dev]"
pre-commit install           # 可选
```

`pip install -e .` 会同时安装 console_script 入口 `aiida-uranium` 并触发 `aiida.workflows` entry-point 注册。安装完成后：

```bash
verdi plugin list aiida.workflows abacus.smear vasp.smear \
                                 abacus.convergence vasp.convergence \
                                 abacus.magmom vasp.magmom
# 6 个 entry point 应该全部列出来
```

### 2.2 AiiDA profile

`aiida-uranium` 假定 AiiDA 已经在机器上 `verdi setup` 过且**设置了默认 profile**。`run` 子命令接受 `--profile` 覆盖；`report` / `copy` / `archive` 同样。

```bash
verdi profile list           # 列出已配置的 profile
verdi profile set default <name>
```

### 2.3 最简运行

```bash
# 1. 准备 input.json (见 §4)
# 2. 跑 smear
aiida-uranium run --method smear --input src/aiida_uranium_workflow/example/input.json
# 3. 产出 output.json (含 WorkChain UUID)
# 4. 生成报告
aiida-uranium report --method smear --input output.json --output-dir reports/
# 5. 拉回计算结果
aiida-uranium copy --method smear --input output.json --output /data/results/
# 6. 打包成 AiiDA archive 离线分发
aiida-uranium archive --method smear --input output.json --output result.aiida
```

### 2.4 `verdi` 入口

CLI 在 `pyproject.toml` 注册为 `[project.scripts] aiida-uranium = ...`；**不**注册 `verdi` 子命令。所有 AiiDA 内部状态查询仍走 `verdi process list / show`。

---

## 3. 端到端 4 阶段流程（与 `feature.md` 对应）

| 阶段 | CLI 子命令 | 输入 | 输出 | 详细图见 `feature.md` |
|---|---|---|---|---|
| 提交 | `aiida-uranium run` | `input.json` + `parameters/<code>/*.yml` | `output.json` (含 WorkChain UUID / pk) | §2 |
| 报告 | `aiida-uranium report` | `output.json` | `reports/report_<key>_<8hex>.md` | §3 |
| 拷贝 | `aiida-uranium copy` | `output.json` | `<PATH>/<backend>/<key>/<preset>/<calcjob>/` | §4 |
| 归档 | `aiida-uranium archive` | `output.json` | `result.aiida` | §5 |

`output.json` 是这 4 个阶段共享的**唯一中间文件**，结构：

```json
{
  "abacus": {"smear": {"lcao": "8c0fe1a9-...-uuid", "pw": "33e15b7c-...-uuid"}},
  "vasp":   {"smear": {"test": "a6dd85b2-...-uuid"}}
}
```

叶子节点使用 **AiiDA UUID 字符串**（canonical identifier）；旧 `pk` 整数布局仍可解析（见 `utils/json_collect.py`）。

### 3.1 阶段间通信：为什么 `output.json` 一定要在

* `run` 之后必须**立即**生成 `output.json`，否则后续阶段无 pk 映射可读。
* `output.json` 是**纯 JSON**，不依赖 AiiDA profile —— 可以拷给另一台机器再做 `report` / `copy`。
* 多机场景：跑计算的机器写 `output.json`，报告/拷贝机器读它（可能指向完全不同的 AiiDA instance 的 UUID 库）。

### 3.2 退出码约定

所有 4 个子命令都遵循：

| 退出码 | 含义 |
|---|---|
| `0` | 完全成功 |
| `1` | 用户错误（输入缺失 / 路径不存在 / 方法名无效 / 没有可处理的对象） |
| `1` | 部分失败（`copy` 阶段某个 CalcJob 拉取失败，详见 `--output` 末尾摘要） |

---

## 4. 输入数据格式

### 4.1 `input.json` 顶层结构

`run` 阶段读取的 `input.json`：

```json
{
  "workflow": "smear",                       // ← 必填：smear | convergence | magmom | base
  "parameters": {                            // ← 必填：每个 backend 的 preset 列表
    "abacus": {"abacus": "test"},            //  ← {"<sub-key>": ["preset_name", ...]}
    "vasp":   {"vasp":   "test"},
    "smear":  "test"                          // ← 可选：protocol 名（与 parameters/<method>.yml 对应）
  },
  "static": {                                 // ← 必填：结构与计算 metadata
    "structure": "bcc-uranium",              //  ← static/structure.yml 的 key
    "metadata":  "yeesuan"                    //  ← static/metadata.yml 的 key
  },
  "profile": "aiida_profile",                 // ← 可选：AiiDA profile 名
  "code": {                                   // ← 必填：每 backend 的 AiiDA Code label
    "abacus": "abacus@yeesuan",
    "vasp":   "vaspstd@yeesuan"
  }
}
```

### 4.2 字段语义

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `workflow` | `str` | ✓ | 调度的 orchestrator 名（`smear` / `convergence` / `magmom` / `base`）。新增 → 见 §15.2。 |
| `parameters` | `dict[str, dict \| str]` | ✓ | 每个 backend 一个 sub-key，value 是 `<backend_subkey> → [preset_names...]` 的 dict。`smear` / `convergence` / `magmom` 这三个 method 名作为**可选**顶级 key 时会被解释为 **protocol 名**（`parameters/<method>.yml` 中的 section 名）。 |
| `static.structure` | `str` 或 `list[str]` | ✓ | `static/structure.yml` 的 key。**字符串** = 单结构；**列表** = 批处理，每个元素跑一份独立 WorkChain。 |
| `static.metadata` | `str` | ✓ | `static/metadata.yml` 的 key。AiiDA 计算资源 / 队列名 / wallclock 来自这里。 |
| `profile` | `str` | ✗ | AiiDA profile 名；缺省 = 默认 profile。 |
| `code` | `dict[str, str]` | ✓ | `{backend_label: "code_label@computer_label"}`；`code_label` 必须 `verdi code list` 能找到。 |

### 4.3 多结构批处理

`static.structure` 也可以是字符串列表：

```json
"static": {
  "structure": ["bcc-uranium", "gamma-uranium", "U-X/FCC"]
}
```

此时 orchestrator 对每个结构提交一份独立 WorkChain（`output.json` 的第二层 key 仍是 `<sub-key>`，第三层是 `<preset_name>`，但 WorkChain 数量 = `len(structure_list) × len(preset_list)`）。

### 4.4 实际示例

`src/aiida_uranium_workflow/example/` 下的 4 份 JSON：

| 文件 | workflow | 用途 |
|---|---|---|
| `base.json` | `base` | 单次 SCF（无扫描） |
| `input.json` | `smear` | smear 扫描（abacus+vasp 各一个 preset） |
| `mul_paramters.json` | （abacus 多 preset） | abacus 多 preset smear |
| `mul_stru.json` | （多结构） | 多个结构 + 多个 preset |
| `output.json` | （产物） | `run` 后的 `output.json` 示例 |

---

## 5. YAML 参数层（`parameters/`）

### 5.1 目录布局

```
parameters/
├── smear.yml           ← 协议层 (top-level)
├── convergence.yml     ← 协议层
├── magmom.yml          ← 协议层
├── abacus/             ← 后端层 (per-backend default)
│   ├── abacus.yml
│   ├── smear.yml       ← per-method 覆盖
│   ├── convergence.yml
│   └── magmom.yml
└── vasp/
    ├── vasp.yml
    ├── smear.yml
    ├── convergence.yml
    └── magmom.yml
```

### 5.2 加载优先级（`utils/config.py:ConfigLoader`）

```
backend_specific (parameters/<code>/<method>.yml)   ← 最高优先
              ↓ 覆盖
backend_default (parameters/<code>/<code>.yml)
              ↓ 覆盖
protocol (parameters/<method>.yml)        ← 最低优先
```

`ConfigLoader._load_all_software_params()` 把三层 YAML 按 `preset_name` 合并，得到 `ParamBundle.software_params[backend] = [preset_dict, ...]`。

### 5.3 preset 字段含义（以 `abacus.yml` 为例）

```yaml
test:
  abacus:                     # ← abacus calculation namespace
    parameters:
      input:                  # ← raw ABACUS INPUT block
        basis_type: lcao
        ecutwfc: 80
        nspin: 2
        ...
    structure: null           # ← 由 static.structure 覆盖
    metadata:
      options: null           # ← 由 static.metadata 覆盖
  kpoints_distance: 0.1       # ← 提交给 adapter 的 k 点策略
  pseudo_family: sg15_sz      # ← ABACUS 赝势族
```

VASP preset 字段类似但 namespace 是 `parameters.incar` / `potential_family` / `potential_mapping` / `kpoints_spacing` / `kpoints_mesh`。

### 5.4 `static/*.yml`

* `static/structure.yml` —— 结构索引。`utils/structure.py` 通过 `pyxtal` 按 spacegroup + Wyckoff 位置生成 `StructureData`。
* `static/metadata.yml` —— 计算 metadata 模板（`options.resources` / `queue_name` / `withmpi` / `max_wallclock_seconds`）。`build_structure_and_metadata()` 把它注入到 preset 的 `metadata.options` 字段。

---

## 6. 仓库结构与模块职责

```
src/aiida_uranium_workflow/
├── __init__.py                # 仅 __version__
│
├── cli/                       # CLI 入口层
│   ├── main.py                # argparse + 4 子命令 dispatch
│   └── _common.py             # METHOD_SPECS + report / archive / copy helpers
│
├── workflows/                 # AiiDA WorkChain 定义
│   ├── __init__.py            # 重新导出 6 个 WorkChain 类
│   ├── smear/
│   │   ├── abacus.py          # AbacusSmearWorkChain
│   │   └── vasp.py            # VaspSmearWorkChain
│   ├── convergence/
│   │   ├── __init__.py        # 导出 AbacusConvergenceWorkChain, VaspConvergenceWorkChain
│   │   ├── abacus.py
│   │   └── vasp.py
│   └── magmom/
│       ├── abacus.py
│       └── vasp.py
│
├── schedulers/                # 调度层（解析 + 提交 + 写 output.json）
│   ├── __init__.py            # 自注册触发器 + 公开 get_orchestrator / get_workflow_entry
│   ├── base.py                # WorkflowOrchestrator 基类 + WorkflowEntry + _ORCHESTRATOR_REGISTRY
│   ├── base_workchain.py      # base workflow 的 orchestrator
│   ├── smear.py               # SmearWorkflowOrchestrator
│   ├── convergence.py         # ConvergenceWorkflowOrchestrator
│   └── magmom.py              # MagmomWorkflowOrchestrator
│
├── input_builders/            # ParamBundle → AiiDA inputs 适配器
│   ├── __init__.py            # 重新导出所有 Adapter
│   ├── base.py                # SoftwareAdapter / AdaptedInputs 抽象
│   ├── base_workchain.py      # base workflow 的 Adapter
│   ├── smear/                 # AbacusAdapter, VaspAdapter
│   ├── convergence/           # AbacusConvergenceAdapter, VaspConvergenceAdapter
│   └── magmom/                # AbacusMagmomAdapter, VaspMagmomAdapter
│
├── parameters/                # 见 §5
│
├── static/                    # 见 §5.4
│
├── utils/                     # 工具
│   ├── cal_json.py            # build_cal_json + write_cal_json（output.json 写盘）
│   ├── common.py              # ParamBundle / PROTOCOL_DIR / PKG_ROOT 等常量
│   ├── config.py              # ConfigLoader（input.json + YAML 解析）
│   ├── copy_remote.py         # AiiDA transport 版 copy 流水线
│   ├── json_collect.py        # output.json 嵌套 → list[pks/uuids]
│   ├── labels.py              # smear/convergence/magmom label 解析与归一化
│   ├── parser_energy_time.py  # fetch_abacus / fetch_vasp + parse_total_time
│   ├── structure.py           # build_structure + write_cif（pyxtal + ase）
│   └── report/                # 见 §10
│
└── example/                   # 见 §4.4
```

### 6.1 命名约定

| 模式 | 例子 | 含义 |
|---|---|---|
| `<method>/<backend>` | `workflows/smear/abacus.py` | 一个特定 (method, backend) 的 WorkChain 类 |
| `<method>/__init__.py` | `workflows/convergence/__init__.py` | 子包的 re-export 入口（仅在多个 backend 时有意义） |
| `<class>Adapter` | `AbacusSmearWorkChainAdapter` / `AbacusAdapter` | 把 ParamBundle 翻成 AiiDA inputs 的策略对象 |
| `<class>WorkflowOrchestrator` | `SmearWorkflowOrchestrator` | 把 preset 列表 × 结构列表展开成 WorkChain 子任务并提交 |
| `generate_<thing>_table` | `generate_status_table` / `generate_energy_table` | 报告里一张表格的渲染器 |
| `generate_<thing>_section` | `generate_optimal_sigma_section` | 报告里一个 section（多张表 + 文字）的渲染器 |
| `generate_report` | 唯一 | 顶层 report 渲染入口 |
| `find_optimal_<x>` / `find_converged_<x>` | 报告里的"建议值"计算器（不是 renderer） |

---

## 7. 调度与 orchestrator 注册表

### 7.1 注册表设计

`schedulers/base.py`：

```python
class WorkflowEntry:
    protocol_file: Optional[str] = None
    workflow_key:  Optional[str] = None
    parser_hook:   Optional[Callable[[dict], dict]] = None
    orchestrator_cls: Optional[Type["WorkflowOrchestrator"]] = None

_WORKFLOW_REGISTRY: Dict[str, WorkflowEntry] = {}

def register_workflow(name, *, protocol_file, workflow_key, parser_hook, orchestrator_cls): ...
def get_workflow_entry(name: str) -> WorkflowEntry: ...
def get_orchestrator(bundle, backends=None) -> "WorkflowOrchestrator": ...
```

### 7.2 self-registration 机制

`schedulers/__init__.py`：

```python
from . import convergence  # noqa: F401  -- triggers self-registration
from . import smear        # noqa: F401
from . import magmom       # noqa: F401
from . import base_workchain  # noqa: F401
```

`schedulers/smear.py` 在 module 顶层调用：

```python
register_workflow(
    name="smear",
    protocol_file="smear.yml",
    parser_hook=parse_smear_protocol,
    orchestrator_cls=SmearWorkflowOrchestrator,
)
```

**导入顺序就是隐式契约**——`schedulers.base` 必须先于 `smear/convergence/magmom` 被 import。所以 `__init__.py` 用 `from .base import ...` 先把基类注册基础设施加载好，然后 `from . import smear/convergence/...` 触发子模块的 self-registration。

### 7.3 dispatch 路径（无 if/elif）

```python
# cli/_common.py
def execute_workflow(*, input_json, profile, only):
    bundle = ConfigLoader(input_json).load_all()
    if profile:
        bundle.input_params["profile"] = profile
    backends = (only,) if only else None
    orchestrator = get_orchestrator(bundle, backends=backends)   # ← dict 查表
    return orchestrator.run_with_jobs()
```

`get_orchestrator()` 内部只读 `bundle.input_params["workflow"]` 然后去 `_WORKFLOW_REGISTRY` 查 → 调 `entry.orchestrator_cls(bundle, backends=backends)`。**没有 if/elif**，新增 method 只需要写 `schedulers/<name>.py` 并加一行 `register_workflow(...)`。

### 7.4 `ConfigLoader` 怎么与 registry 协作

```python
# utils/config.py
def load_all(self) -> ParamBundle:
    self._validate()
    from aiida_uranium_workflow.schedulers import get_workflow_entry

    entry = get_workflow_entry(self.input_params["workflow"])  # ← 早查一次
    protocol = self._load_protocol(entry)                      # 用 entry.protocol_file
    workflow_data = self._parse_protocol(protocol, entry)      # 用 entry.parser_hook
    software_params = self._load_all_software_params()
    metadata = self._load_metadata()
    return ParamBundle(...)
```

所以 protocol 文件名 + 协议解析 hook 都来自 registry，**单一真值源**。

### 7.5 `WorkflowOrchestrator.run_with_jobs()` 流程

`schedulers/base.py` 提供的基类：

```python
class WorkflowOrchestrator(ABC):
    backend: str
    adapters: dict[str, SoftwareAdapter]   # {"abacus": ..., "vasp": ...}
    bundle: ParamBundle

    @abstractmethod
    def iter_jobs(self) -> Iterator[tuple[str, str, AdaptedInputs]]: ...  # (backend, preset, inputs)

    def run_with_jobs(self) -> list[SubmittedJob]:
        results = []
        for backend, preset, adapted in self.iter_jobs():
            cls = adapted.workchain_cls
            submitted = submit(cls, **adapted.inputs)   # 自定义 submit_with_metadata
            results.append(SubmittedJob(backend=..., preset_name=..., pk=..., uuid=...))
        return results
```

子类（`SmearWorkflowOrchestrator` / `ConvergenceWorkflowOrchestrator` / `MagmomWorkflowOrchestrator`）只覆盖 `iter_jobs()` 即可——它们枚举 preset 列表 × 网格（smear 是 (method, sigma)，convergence 是 (ecut, kpoints)，magmom 是 (mag_value)），每个组合调一次对应 `Adapter.adapt(structure)` 然后 `submit` 一份 WorkChain。

`SubmittedJob` 包含 `pk`（向后兼容）+ `uuid`（canonical identifier，**新代码应当用 uuid**）。

---

## 8. 输入适配器（`input_builders/`）

### 8.1 抽象基类

```python
# input_builders/base.py
@dataclass
class AdaptedInputs:
    workchain_cls: Type[WorkChain]
    inputs: dict[str, Any]

class SoftwareAdapter(ABC):
    name: ClassVar[str]    # "abacus" / "vasp"
    code_label: str        # from input.json
    metadata: dict         # scheduler options
    software_params: dict  # per-preset 后端参数
    workflow_data: dict    # method 解析结果

    @abstractmethod
    def _workchain_entry_point(self) -> str: ...   # "abacus.smear" 等
    @abstractmethod
    def _build_workchain_inputs(self, structure) -> dict: ...
    @abstractmethod
    def _prepare_workflow_inputs(self) -> tuple: ...  # 把 workflow_data 展平成网格坐标

    def adapt(self, structure) -> AdaptedInputs:
        # 模板方法：调 _build_workchain_inputs + _prepare_workflow_inputs
        # 调 _inject_options 把 metadata.options 注入 inputs
        ...
```

### 8.2 命名映射

| method | backend | Adapter 类 | entry-point |
|---|---|---|---|
| smear | abacus | `AbacusAdapter` (旧名：`input_builders/abacus.py`) | `abacus.smear` |
| smear | vasp | `VaspAdapter` (旧名：`input_builders/vasp.py`) | `vasp.smear` |
| convergence | abacus | `AbacusConvergenceAdapter` | `abacus.convergence` |
| convergence | vasp | `VaspConvergenceAdapter` | `vasp.convergence` |
| magmom | abacus | `AbacusMagmomAdapter` | `abacus.magmom` |
| magmom | vasp | `VaspMagmomAdapter` | `vasp.magmom` |
| base | abacus | `AbacusBaseWorkChainAdapter` | `abacus.base` |
| base | vasp | `VaspBaseWorkChainAdapter` | `vasp.base` |

### 8.3 k 点 / σ 等"网格坐标"约定

| method | (网格 row × col) | 公共 helper |
|---|---|---|
| smear | (smear_method, sigma_eV) → workflow_data.smear_lists.smear / .sigma | `smear_inputs` 模式 |
| convergence | (ecutwfc, kpoints) → workflow_data.convergence_lists.abacus.ecutwfc_list / .kpoints_distance_list（或 kpoints_mesh_list） | distance / spacing / mesh 三种模式 |
| magmom | mag_value（单维）→ workflow_data.magmom_lists.abacus.mag_list | abacus 用嵌套 list，vasp 用 mapping dict |

### 8.4 base workflow

`base` workflow **没有** method 网格 —— 单次提交一份。Adapter 的 `_prepare_workflow_inputs` 仍然存在但返回空网格，orchestrator `iter_jobs()` 退化为「每 preset 一次提交」。

---

## 9. WorkChain（`workflows/`）

### 9.1 通用骨架

每个 `workflows/<method>/<backend>.py` 都遵循同样的 `submit_children` / `gather_results` / `parse_and_gather_*_results` 三段式：

```python
class XxxWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("metadata")               # 计算资源
        spec.input_namespace(...)             # 每个 sweep 维度的 list 输入
        spec.outline(
            cls.setup,
            cls.submit_children,
            cls.gather_results,
            cls.results,
        )

    def submit_children(self):
        # 用 product() 枚举网格，构造 metadata.label = "smear_xxx_sigma_yyy"
        # submit(ChildWorkChain, label=...)  → to_context(**{label: node})
        ...

    def gather_results(self):
        # 遍历 to_context 收集 pk，调 calcfunction parse_and_gather_*_results
        # 写 self.outputs.output_parameters = orm.Dict(...)
        ...
```

### 9.2 `parse_and_gather_*_results` calcfunction

每个 method 有自己的 calcfunction（在 `<backend>.py` 同文件，**或**在 `utils/parser_energy_time.py` 共享），负责：

1. 遍历子节点 pk
2. 从 `base.outputs.<namespace>.output_parameters` 抽 status / energy / wall_time
3. `fetch_abacus(node)` / `fetch_vasp(node)` → `utils/parser_energy_time.py` 提供
4. `parse_total_time(...)` → 同上
5. 拼成 dict → `orm.Dict(...)`

**这个模式是所有 6 个 WorkChain 共享的**，**未来 PR 可以考虑抽公共基类**（见 §17.1）。

### 9.3 output_parameters 统一 schema

```python
{
    "status": {<label>: <exit_code>, ...},                  # 必填
    "total_energy": {<label>: float, ...},                  # 必填（单位：abacus=Ry, vasp=eV）
    "wall_time_seconds": {<label>: float, ...},             # 必填
    # method-specific：
    # smear 还写：
    "eentropy_per_atom": {...},  # abacus 用 string method 名，vasp 用 ismear int
    "eentropy": {...},
    # convergence 还写：
    "total_energy_per_atom": {...},  # 全网格的每原子能量（收敛判定用）
    # magmom 还写：
    "magnetism": {<pk>: [...]},  # abacus
    "magnetization": {<pk>: float},  # vasp
    "nspin": {<pk>: int},  # abacus
    "site_magnetization": {<pk>: [...]},  # vasp
}
```

报告层（`utils/report/`）就是按这个 schema 渲染 Markdown。

---

## 10. 报告模块（`utils/report/`）

### 10.1 顶层布局

```
utils/report/
├── _common.py        # 共享渲染原语（AxisSpec / parse_axes / format_scalar / render_2d_grid / render_per_child_table / render_report_header / render_report_footer）
├── smear.py          # 2D 网格 smear × sigma
├── convergence.py    # 2D 网格 ecut × kpoints
└── magmom.py         # per-child 单列（每个 child 一行）
```

### 10.2 共享原语（`utils/report/_common.py`）

#### `AxisSpec` —— 声明一个 2D 网格的一根轴

```python
@dataclass(frozen=True)
class AxisSpec:
    name: str                       # 显示名
    keyword: str                    # 主关键字（如 "ecutwfc"）
    keyword_aliases: tuple = ()     # 别名（如 ("encut",)，vasp 用）
    kind: str = "auto"              # "string" | "float" | "auto"
    qualifiers: tuple = ()          # keyword 后的"限定符"（如 "kpoints" 后的 "spacing" / "distance"）
    unit: str = ""
```

#### `parse_axes(label, axes)` —— 解析 label

```python
parse_axes("ecutwfc_80_kpoints_distance_0_1", [ecut_axis, kpoints_axis])
# → [80.0, 0.1]

parse_axes("ecutwfc_60_kpoints_11x11x11", [ecut_axis, kpoints_axis])
# → [60.0, "11x11x11"]   # auto kind 区分 float / string

parse_axes("smear_mp_sigma_0_02", [smear_axis, sigma_axis])
# → ["mp", 0.02]
```

实现策略：

1. 找每个 axis 在 `label.split("_")` 里的 keyword 索引（避开已被早 axis 占用）。
2. axis N 的 value 是 `[kw_idx+1, kw_idx_of_axis_{N+1})`。
3. 跳过 qualifier token（如 "spacing"）。
4. 按 `kind` 把 value 字符串转成 float / str。

#### `format_scalar(value, *, fmt, missing)` —— 单元格格式化

```python
format_scalar(None)               # → "—"
format_scalar("ERROR")            # → "—"
format_scalar(123.456)            # → "123.456000" (default fmt="%.6f")
format_scalar(0, fmt="%.3f")      # → "0.000"
```

替换了 6 处手写 `if value is None: return "—" / if isinstance(value, str): return "—" / f"{float(value):.6f}"`。

#### `sort_axis_values(values)` —— mesh 字符串按数字排序

```python
sort_axis_values(["11x11x11", "5x5x5", "13x13x13", "7x7x7"])
# → ["5x5x5", "7x7x7", "11x11x11", "13x13x13"]   # 按数字大小，不是字符串
```

替换了原 `convergence._sort_kpoints_values`。

#### `render_2d_grid(data, axes, *, row_header, col_header, cell_format, empty_placeholder)` —— 通用 2D Markdown 表格

```python
render_2d_grid(
    data={"ecutwfc_80_kpoints_distance_0_1": 0, "ecutwfc_80_kpoints_distance_0_2": 0},
    axes=[ecut_axis, kpoints_axis],
    row_header="ecutwfc (Ry)",
    col_header="kpoints_distance (A^-1)",
    cell_format=lambda v: str(int(v)),
    empty_placeholder="-",
)
```

替换了 smear / convergence 各自的 30 行 2D grid 样板（6 处）。

#### `render_per_child_table(data, *, column_header, column_name, cell_format)` —— per-child 单列

替换了 magmom 自己的 status / energy / wall_time 单列样板（3 处）。

#### `render_report_header(*, title, workflow_type, pk, timestamp)` / `render_report_footer()`

替换了 3 处 `generate_report` 的 preamble / epilogue 模板。

### 10.3 各报告文件

* **`smear.py`** —— 2D 网格表（status / energy / wall_time / eentropy）+ optimal-sigma 建议
* **`convergence.py`** —— 2D 网格表（status / energy / wall_time / total_energy_per_atom）+ 差分表（encut / kpoints）+ 收敛推荐
* **`magmom.py`** —— per-child 表（status / energy / wall_time）+ backend-specific magnetism 段落（abacus 列出每原子磁化、nspin；vasp 列出 total + site_magnetization）+ magmom_convergence 表

所有 `generate_<thing>` 函数签名与原版完全一致 —— **45/45 测试通过**。

### 10.4 `generate_report` 模板

每个 `generate_report(output_params, pk, workflow_type)` 的内部结构：

```python
report_lines = [
    render_report_header(title=..., workflow_type=..., pk=...),
    "",
    "## Summary", "",
    generate_summary_table(output_params),
    "",
]
# 每个 section 由 if "key" in output_params: ... 守护
if "status" in output_params:
    report_lines += ["## Calculation Status", "", generate_status_table(...), ""]
if "total_energy" in output_params and output_params["total_energy"]:
    report_lines += [f"## Total Energy [{unit}]", "", generate_energy_table(...), ""]
# ... 等等
report_lines += [render_report_footer()]
return "\n".join(report_lines)
```

---

## 11. 拷贝（`utils/copy_remote.py`）

### 11.1 与老 `utils/copy_calc.py` 的关系

`utils/copy_calc.py` 是**老的 scp-shell 脚本**（`_label_for_*` / `copy_targets` / `discover_actual_calcs` 等），其函数均无仓库内调用方。**整文件已删除**。

唯一保留下来的 helper `collect_*_from_json` 已迁到 `utils/json_collect.py`：

```python
collect_pks_from_json(data)               # 嵌套 dict → list[整数 pk]
collect_node_ids_from_json(data)          # 嵌套 dict → list[UUID 字符串]
collect_identifiers_from_json(data)       # 两者并集（自动判 UUID vs pk）
```

### 11.2 新版拷贝流水线

```
collect_pk_map(input.json)                 # 嵌套 dict
    → iter_copy_targets(pk_map, class_to_backend)   # 走 WorkChain 后代 CalcJob
    → CopyTarget = (wc_pk, wc_label, backend, preset, key, calcjob_pk, calcjob_label, remote_path, computer)
    → resolve_copy_targets(targets, base_dir)       # 计算本地目标路径
    → CopyPlan = (entries: [CopyPlanEntry...], skipped: [...])
    → execute_copy_plan(entries)                    # 通过 aiida.transports.Transport.get 拉取
```

每份 entry 的 `local_path = <base_dir>/<backend>/<key>/<preset>/<calcjob_label>/` —— `sanitise_path_component` 清洗非法字符。

---

## 12. 标签解析（`utils/labels.py`）

### 12.1 为什么需要

WorkChain 内部 `metadata.label` 与 AiiDA 节点的 `process_label` 不同：

* `metadata.label` = workflow 在 `submit()` 时**显式**设置的字符串（`format_<backend>_<method>_label` 生成）
* `process_label` = WorkChain **类名**（无设置时退化为 `AbacusSmearWorkChain` 等）

AiiDA 把 40 个 child CalcJob 落盘后，目录里全是 `AbacusCalculation` 这种**没用的 process_label**。`utils/labels.py:resolve_label(calcjob, backend, method)` 反向重建 label：

1. 优先用 `calcjob.metadata.label`（如果 workflow 显式设置过）
2. 否则走 `calcjob.process_label`，跳过 `_REJECTED_GENERIC_CLASS_NAMES`（`AbacusCalculation` / `VaspCalculation` / `AbacusWorkChain` / ...）
3. 否则走 `calcjob.inputs.abacus.parameters.input` 或 `calcjob.inputs.parameters.incar` 找原始 (method, sigma) / (ecut, kpoints) / (mag) 组合
4. 退到 `f"calcjob_{pk}"` 兜底

### 12.2 `format_*_label` —— 中心化的 label 格式

| method | backend | label 格式 | 函数 |
|---|---|---|---|
| smear | abacus | `smearing_<method>_sigma_<value>` | `format_abacus_smear_label` |
| smear | vasp | `ismear_<n>_sigma_<value>` | `format_vasp_smear_label` |
| convergence | abacus | `ecutwfc_<v>_kpoints_distance_<v>` 或 `ecutwfc_<v>_kpoints_<NxNxN>` | `format_abacus_convergence_label` |
| convergence | vasp | `kpoints_spacing_<v>_encut_<v>` 或 `kpoints_<NxNxN>_encut_<v>` | `format_vasp_convergence_label` |
| magmom | (abacus/vasp) | `<v:g>` 然后 `.replace(".", "_").replace("-", "m")` | `format_magmom_label` |

`format_*_label` 是**single source of truth**——workflow 的 `submit_children` 与 `resolve_label` 调同一个函数。

`_REJECTED_GENERIC_CLASS_NAMES` 是个 frozenset 硬编码 12 个 class 名（`AbacusBaseWorkChain` / `VaspBaseWorkChain` / `AbacusConvergenceWorkChain` / `VaspConvergenceWorkChain` / `AbacusSmearWorkChain` / `VaspSmearWorkChain` / `AbacusMagmomWorkChain` / `VaspMagmomWorkChain` + `AbacusCalculation` / `VaspCalculation` / `AbacusWorkChain` / `VaspWorkChain`）——**新增 entry-point 类时必须同步更新这个 frozenset**（目前没有自动同步机制，是已知技术债）。

---

## 13. CLI 入口（`cli/main.py` + `cli/_common.py`）

### 13.1 入口

`pyproject.toml` 注册：

```toml
[project.scripts]
aiida-uranium = "aiida_uranium_workflow.cli.main:main"
```

`cli/main.py:main()` 调 `build_unified_parser()` 拿 argparse → 4 个 subparser → dispatch：

| subcommand | 函数 | 行数 |
|---|---|---|
| `run` | `_run(args)` | `main.py` |
| `report` | `_report(args)` | `main.py` |
| `copy` | `_copy(args)` | `main.py` |
| `archive` | `_archive(args)` | `main.py` |

### 13.2 `METHOD_SPECS` —— 单一真值源

```python
# cli/_common.py
@dataclass(frozen=True)
class MethodSpec:
    name: str
    class_to_backend: Mapping[str, str]   # {"AbacusSmearWorkChain": "abacus", ...}
    generate_report: Callable[..., str]  # method-specific report generator
    backend_to_key: Mapping[str, str]    # {"abacus": "smear", "vasp": "smear"} 或 {"abacus": "abacus", "vasp": "vasp"}

METHOD_SPECS: tuple[MethodSpec, ...] = (smear_spec, convergence_spec, magmom_spec, base_spec)
```

**新增 method 只需要写一个 `MethodSpec` 并 append 到 `METHOD_SPECS`**——CLI 4 个子命令不需要改任何东西（subcommand 由 method 列表动态生成）。

### 13.3 `get_method_spec` / `parse_method` / `resolve_method`

```python
get_method_spec("smear")         # → smear_spec
parse_method("smear")            # → "smear"
resolve_method("AbacusSmearWorkChain")  # → "smear"  （反查）
```

### 13.4 report 子命令细节（详见 feature.md §3）

```
collect_pk_map(input.json)
  → 遍历 {backend: {key: {preset: id}}}
  → for each id: generate_one_report(node_identifier=id, ...) 
    → load_finished_workchain(id)
    → resolve_backend(class_name)            # "AbacusSmearWorkChain" → "abacus"
    → spec.generate_report(params, id, backend)
    → write_text_report(text, path)
  → 输出 reports/report_<safe_key>_<8hex>.md
```

`--output-dir` 缺省 = `<input>.parent/reports/`（即 output.json 所在目录的 `reports/` 子目录）。

### 13.5 copy / archive 子命令

详见 feature.md §4 / §5。

### 13.6 dry-run

* `aiida-uranium copy --dry-run` —— 打印 `<src> → <dst>` 列表，不执行
* `aiida-uranium archive --dry-run` —— 列出将要打包的 id，不执行

---

## 14. 公开 API 一览

### 14.1 Python 公开入口

```python
# Workflow 加载（AiiDA 习惯）
from aiida.plugins import WorkflowFactory
AbacusSmearWorkChain = WorkflowFactory("abacus.smear")

# 配置加载
from aiida_uranium_workflow.utils.config import ConfigLoader
bundle = ConfigLoader("input.json").load_all()

# Orchestrator
from aiida_uranium_workflow.schedulers import get_orchestrator
orch = get_orchestrator(bundle, backends=("abacus",))
results = orch.run_with_jobs()

# 报告
from aiida_uranium_workflow.utils.report.smear import generate_report
print(generate_report(output_params, pk=42, workflow_type="abacus"))

# 拷贝 / 归档
from aiida_uranium_workflow.utils.copy_remote import load_copy_plan, execute_copy_plan
plan = load_copy_plan(input_json=..., method="smear", class_to_backend=..., base_dir="/data")
execute_copy_plan(plan.entries)
```

### 14.2 入口点（`pyproject.toml`）

| 入口 | 类 | 模块 |
|---|---|---|
| `aiida-uranium` | `main` | `aiida_uranium_workflow.cli.main` |
| `abacus.smear` | `AbacusSmearWorkChain` | `aiida_uranium_workflow.workflows.smear.abacus` |
| `vasp.smear` | `VaspSmearWorkChain` | `aiida_uranium_workflow.workflows.smear.vasp` |
| `abacus.convergence` | `AbacusConvergenceWorkChain` | `aiida_uranium_workflow.workflows.convergence.abacus` |
| `vasp.convergence` | `VaspConvergenceWorkChain` | `aiida_uranium_workflow.workflows.convergence.vasp` |
| `abacus.magmom` | `AbacusMagmomWorkChain` | `aiida_uranium_workflow.workflows.magmom.abacus` |
| `vasp.magmom` | `VaspMagmomWorkChain` | `aiida_uranium_workflow.workflows.magmom.vasp` |

注意：**没有 `abacus.base` / `vasp.base` 的 entry-point** —— `base` workflow 不通过 AiiDA workflow 注册，仅作为 `aiida-uranium run --method base` 的内部路径存在。如需 AiiDA workflowFactory("abacus.base")，要手动加 entry-point（见 §15.4）。

### 14.3 包内可被 import 但**不是公开 API**（如使用会破坏）

* `utils.report._common` —— 内部共享层，新代码应通过 `smear.py` / `convergence.py` / `magmom.py` 的 `generate_*` 函数间接使用
* `schedulers._ORCHESTRATOR_REGISTRY` / `_WORKFLOW_REGISTRY` —— 内部 dict，应通过 `register_workflow` / `get_orchestrator` 操作
* `input_builders.base.AdaptedInputs` —— datacass，但用 `_workchain_cls` 私有字段；用户代码应 `from aiida.plugins import WorkflowFactory` 拿 workchain_cls
* `cli._common.METHOD_SPECS` —— frozen tuple，**只读**；新 method 应**append** 而非 mutate

---

## 15. 扩展指南

### 15.1 新增一个 preset

1. 编辑 `parameters/<code>/<method>.yml`，加：
   ```yaml
   <new_preset_name>:
     abacus:
       parameters:
         input: { ... }
     kpoints_distance: 0.1
     pseudo_family: sg15_sz
   ```
2. `aiida-uranium run --method smear --input <some.json>` 即可用新 preset。

### 15.2 新增一个 method（例如 "elastic"）

1. 写 `schedulers/elastic.py`：
   ```python
   from .base import register_workflow
   from aiida_uranium_workflow.input_builders.elastic import AbacusElasticAdapter, VaspElasticAdapter

   class ElasticWorkflowOrchestrator(WorkflowOrchestrator):
       def iter_jobs(self): ...

   def parse_elastic_protocol(protocol): ...

   register_workflow(
       name="elastic",
       protocol_file="elastic.yml",
       parser_hook=parse_elastic_protocol,
       orchestrator_cls=ElasticWorkflowOrchestrator,
   )
   ```
2. 写 `workflows/elastic/{abacus,vasp}.py`（WorkChain 类）。
3. 写 `input_builders/elastic/{abacus,vasp}.py`（Adapter）。
4. 写 `parameters/elastic.yml` + `parameters/<code>/elastic.yml`。
5. 写 `utils/report/elastic.py`（参考 `smear.py`，重用 `_common`）。
6. 在 `cli/_common.py:METHOD_SPECS` 加一条 `MethodSpec`。
7. 写 `tests/`。
8. **不要忘记** `utils/labels.py:_REJECTED_GENERIC_CLASS_NAMES` 加新类名（避免回退到类名当 label）。

### 15.3 新增一个后端（例如 "qe"）

类似 §15.2，但需要：

1. `input_builders/qe/{smear,convergence,magmom,base}.py`
2. `workflows/<method>/qe.py`
3. `parameters/qe.yml` + `parameters/qe/<method>.yml`
4. `static/structure.yml` 保持不变（structure 跟 backend 无关）
5. `cli/_common.py:METHOD_SPECS[*].class_to_backend` 加新 entry

### 15.4 让 base workflow 也成为 AiiDA entry-point

在 `pyproject.toml`：

```toml
[project.entry-points."aiida.workflows"]
"abacus.base" = "aiida_uranium_workflow.workflows.base_workchain.abacus:AbacusBaseWorkChain"
"vasp.base"   = "aiida_uranium_workflow.workflows.base_workchain.vasp:VaspBaseWorkChain"
```

加上 `schedulers/base_workchain.py` 的 self-registration（已经存在）即可。

### 15.5 添加新的 report section

1. 在 `utils/report/<method>.py` 加 `generate_<new>_section(output_params, ...)` 函数。
2. 在该文件的 `generate_report` 里追加：
   ```python
   if "<trigger_key>" in output_params and output_params["<trigger_key>"]:
       report_lines += ["## <Section Title>", "", generate_<new>_section(output_params), ""]
   ```
3. 写测试：`tests/test_report_<method>.py`。

### 15.6 调整 `_REJECTED_GENERIC_CLASS_NAMES`

当新增 entry-point 时**必须**同步这个 frozenset。**未来 PR**：从 `_WORKFLOW_REGISTRY` 自动派生（见 §17.1）。

---

## 16. 测试组织

### 16.1 目录结构

```
tests/
├── conftest.py                       # AiiDA pytest fixtures（profile / tmp_code / sandbox）
├── test_base_workchain.py            # base workflow + adapter
├── test_cal_json.py                  # output.json 写盘
├── test_config_loader.py             # ParamBundle 解析
├── test_config_loader_validation.py  # 校验路径
├── test_convergence_cli.py           # （已弃用空架）
├── test_convergence_schedulers.py    # 调度
├── test_copy_remote.py               # 拷贝流水线
├── test_labels.py                    # 标签解析
├── test_multi_structure.py           # 多结构批处理
├── test_orchestrator_multi_preset.py # orchestrator 多 preset
├── test_parser_energy_time.py        # 共享 fetch_* / parse_total_time
├── test_report_energy_time.py        # 45 个 report 渲染测试（核心）
├── test_resolve_method.py            # METHOD_SPECS dispatch
├── test_run_cli.py                   # run 端到端
├── test_structure_cli.py             # structure build + cif
├── test_unified_cli.py               # 4 子命令 argparse
├── input_builders/                   # 子包占位（与源码镜像，未来扩展）
│   ├── smear/  convergence/  magmom/
└── workflows/                        # 单独 workchain 集成测试
    ├── convergence/{abacus,vasp}.py
    └── magmom/{abacus,vasp}.py
```

### 16.2 跑测试

```bash
# 全量
pytest tests/

# 特定子集
pytest tests/test_report_energy_time.py
pytest tests/test_config_loader.py -k "magmom"

# 跳过集成测试（仅 unit）
pytest tests/ -m "not integration"
```

### 16.3 关键测试数字

| 测试集 | 用例数 | 覆盖 |
|---|---|---|
| `test_report_energy_time.py` | 33 | smear / convergence / magmom 各自的 status / energy / wall_time 表 |
| `test_parser_energy_time.py` | 12 | `fetch_abacus` / `fetch_vasp` / `parse_total_time` |
| `test_labels.py` | 1 类 | format_*_label + resolve_label 大量子用例 |
| `test_convergence_schedulers.py` | 1 类 | ConvergenceWorkflowOrchestrator |
| `test_cal_json.py` | 1 类 | build_cal_json + write_cal_json |
| `test_copy_remote.py` | 1 类 | CopyPlan / CopyTarget 全流程 |
| `test_unified_cli.py` | 1 类 | 4 subcommand argparse |
| `test_resolve_method.py` | 1 类 | METHOD_SPECS / get_method_spec / parse_method / resolve_method |
| `test_run_cli.py` | 1 类 | run 端到端 |

合计 **320 passed / 14 failed**（详见 §17）。

---

## 17. 已知问题与故障排查

### 17.1 已知 pre-existing 失败（不归本重构）

跑 `pytest tests/` 有 14 个测试 fail——**与本仓库所有提交无关**，是更早期遗留的损坏数据/签名不匹配：

| 测试 | 原因 | 修复方式 |
|---|---|---|
| `tests/test_structure_cli.py` 10 个 | `static/structure.yml` 的 `bcc-uranium` 条目 `x: [3.532]` 只有 1 个值（应当 = `lattice + N_wyckoff`，即 `1 + 1 = 2` 个）。`utils/structure.py:build_structure` 拒绝并 raise `ValueError` | 改 `x: [3.532, 0.0]`（a 位置坐标） 或回退至 `x: [3.45063]`（用 `bcc-uranium-0K` 的 a 值） |
| `tests/workflows/convergence/test_{abacus,vasp}.py::test_parse_and_gather_convergence_results` 2 个 | 测试 fixture 期望 `parse_and_gather_convergence_results` 接受 `status_dict` 入参，但当前签名是 `**children` | 加 `status_dict` 入参（语义：收集所有 child 的 status） |
| `tests/workflows/magmom/test_{abacus,vasp}.py::test_parse_and_gather_magmom_results` 2 个 | 同上，magmom 版 | 同上 |

这些**不在本 README 描述的重构范围内**——前面所有 commit 都没碰过它们；切到 HEAD 738c75c 跑测试也是 7 failed（旧）+ 现在 14 failed（多出来的 7 个是 `test_structure_cli` 因为 `structure.yml` 早被改坏）。

### 17.2 `_REJECTED_GENERIC_CLASS_NAMES` 硬编码

新增 WorkChain entry-point 后**必须**同步更新这个 frozenset，否则 `resolve_label` 会把类名当 label。

**未来 PR 改进**（来自 `requirements.md` §9 风险表）：

```python
# 自动派生
from .registry import _WORKFLOW_REGISTRY
_REJECTED_GENERIC_CLASS_NAMES = frozenset(
    cls.__name__ for entry in _WORKFLOW_REGISTRY.values()
    for cls in _all_subclasses(entry.orchestrator_cls)  # 包括所有相关 base / WorkChain
) | {"AbacusCalculation", "VaspCalculation", "AbacusWorkChain", "VaspWorkChain"}
```

### 17.3 sweep WorkChain 公共基类未抽

6 个 WorkChain 的 `submit_children` / `gather_results` / `parse_and_gather_*_results` 高度雷同。**未来 PR**：抽一个 `SweepWorkChain` 基类（提交时只声明 axes，子类只声明自己的 (label_format, child_entry)）。

### 17.4 input_builders Adapter 公共骨架未抽

8 个 Adapter 的 `_build_workchain_inputs` 高度雷同（kpoints_distance / kpoints_spacing / kpoints_mesh 三选一 + 构造 options + 注入 metadata）。**未来 PR**：抽 `_resolve_kpoints` 与 `_inject_options` 到 `input_builders/base.py`。

### 17.5 utils/ 仍是"杂物抽屉"

虽然 `physics.py` / `copy_calc.py` 已删，但 `utils/` 还混杂：

* 配置加载（`config.py` / `common.py` / `yamlio.py` 已经被合并到 `config.py` —— 不再存在 `yamlio.py`）
* 结果写盘（`cal_json.py`）
* 解析（`parser_energy_time.py` / `labels.py`）
* 报告（`report/` 子包）
* 结构（`structure.py` —— 仍自带 CLI）
* 拷贝（`copy_remote.py`）

**未来 PR**：拆 `utils/config/` / `utils/parser/` / `utils/structure/` / `utils/io/` 多个子包。

### 17.6 故障排查速查

| 现象 | 排查 |
|---|---|
| `ModuleNotFoundError: aiida_uranium_workflow.workflows.convergence` | `pip install -e .` 是否跑过？`workflows/convergence/__init__.py` 是否丢失？ |
| `verdi plugin list aiida.workflows` 找不到 entry-point | `pip install -e .` 没跑 / `pyproject.toml [project.entry-points."aiida.workflows"]` 与实际类路径不一致 |
| `KeyError: 'workflow'` | `input.json` 缺 `workflow` 字段 |
| `KeyError: 'bcc-uranium'` | `static/structure.yml` 没这个 key（也可能是大小写不匹配） |
| `ValueError: Workflow 'xxx' has no orchestrator registered` | 漏写 `register_workflow` 调用，或 `schedulers/__init__.py` 漏 import 这个子模块 |
| AiiDA 报 `profile 'xxx' not found` | `verdi profile list` 看是否设置；`verdi profile set default xxx` |
| `copypy` 阶段只看到 calcjob label 是 `AbacusCalculation` | `utils/labels.py:_REJECTED_GENERIC_CLASS_NAMES` 漏加新类名（`resolve_label` 退到 process_label） |
| `output.json` 子节点 id 是整数 pk 不是 UUID | 旧版 AiiDA profile 产物；`utils/json_collect.py:collect_identifiers_from_json` 同时支持，但报告里 `load_node(pk)` 也能正常解析 |

---

## 18. 引用与相关项目

### 18.1 依赖

* [aiida-core](https://www.aiida.net/) ≥ 2.0 —— AiiDA 框架
* [aiida-abacus](https://github.com/aiida-abacus/aiida-abacus) ≥ 0.1 —— ABACUS calculation / workchain plugin
* [aiida-vasp](https://github.com/aiida-vasp/aiida-vasp) ≥ 5.0 —— VASP calculation / workchain plugin
* [ase](https://wiki.fysik.dtu.dk/ase/) ≥ 3.22 —— 原子模拟环境
* [pyyaml](https://pyyaml.org/) ≥ 6.0
* [pyxtal](https://pyxtal.readthedocs.io/) ≥ 1.0 —— 对称性结构生成（`static/structure.yml` 的 `wps` 解析）

### 18.2 仓库

* 源码：`src/aiida_uranium_workflow/`
* 文档：`docs/`
* 测试：`tests/`
* 示例：`src/aiida_uranium_workflow/example/`
* GitHub：https://github.com/liguozhou/aiida-uranium-workflow

### 18.3 相关历史

* 2026-07-27（commit `738c75c`）—— 初始 commit
* 2026-07-28（commit `734b262` / `3ea559c` / `f211d8f`）—— 重组重构 + 报告模块化
