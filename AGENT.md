# AGENT.md — aiida-uranium-workflow 开发规范

本文件是仓库的"宪法"：任何后续开发（人 or AI agent）在修改本仓库前都应阅读并遵守。
它定义项目定位、架构约定、命名规范、参数体系、测试要求和变更纪律。**与代码矛盾的文档以本文为准，并同步修正文档。**

## 1. 项目定位

AiiDA 工作流编排库：用统一 JSON 驱动 ABACUS / VASP / FLEUR 上的 DFT 计算
（smear / convergence / magmom / banddos / relax / elastic / eos / phonopy），
产出报告、图表、归档与跨代码对比指标。**是"编排层"不是"计算引擎"**——具体 DFT
计算由 aiida-abacus / aiida-vasp / aiida-fleur / aiida-phonopy 插件完成，本库包装它们。

## 2. 核心原则（不可违反）

1. **单源参数控制**：计算参数只存在于 `parameters/` 的 YAML preset 中；任何参数缺失必须
   **报错**，禁止静默 fallback（`utils/config.py` 已有此语义，保持）。
2. **Entry point 命名空间**：本库自定义 WorkChain 一律注册为
   `uranium.<method>.<backend>`；**永远不要**用裸 `abacus.*` / `vasp.*` / `fleur.*`
   （那是插件的命名空间，会冲突）。插件入口（`abacus.base`、`fleur.scf`、`vasp.v2.vasp` 等）
   通过 `WorkflowFactory` 引用，不注册、不改名。
3. **薄包装**：自定义 WorkChain 只做编排（多步、收集、calcfunction 包装输出），物理参数
   交给插件。能用插件 WorkChain 直接完成的（如 relax）就不写自定义 WorkChain。
4. **纯函数与 AiiDA 解耦**：数值逻辑（能带比较、弹性拟合、提取、报告排版）写成不依赖
   AiiDA profile 的纯函数（numpy/pymatgen），放在 `utils/` 下，可单测；AiiDA 只在
   workflows / input_builders / schedulers / cli 层出现。
5. **calcfunction 包装输出**：WorkChain 不能直接输出新建的 Data node；必须用
   `@calcfunction` 包装；calcfunction 输入用单个 JSON 安全的 `orm.Dict`，禁止把 numpy 数组
   塞进 `orm.List`（会扁平化并触发 "truth value of an array" 校验错误）。
6. **对称与可复现**：比较/拟合指标对参数交换对称（A↔B 结果不变）；随机性只允许出现在
   明确标注处。
7. **数据根（data root）机制**：`parameters/` 与 `static/` 不在代码里硬编码，运行时解析
   「数据根」——一个同时含 `parameters/` 和 `static/` 的目录。优先级：
   `--data-dir DIR` / `ConfigLoader(data_root=)` → `$AIIDA_URANIUM_DATA` →
   从 input.json 向上找最近的含 `parameters/` 的目录 → 包内兜底。
   **本机活跃数据根是 `aiida-uranium-scripts/`**（`.../test/<method>/input.json` 的祖先），
   包内 `src/aiida_uranium_workflow/{parameters,static}` 只作**参考 / 测试用例**（顺带在
   找不到数据根时兜底，不在活跃路径上）：**改参数请改数据根那份**，改包内那份不生效
   （解析器会在包内文件比数据根新时告警，见 `snapshot_drift`）。
   实现见 `utils/common.py:resolve_data_root` / `snapshot_drift`。
8. **`parameters/` 不同步到远端仓库**：数据根里的 preset、`static/*.yml`（队列名、结构表）
   **不纳入 git 提交/推送**——本地改动原样留在开发机上，**除非用户当次明确说明要同步**。
   代码与文档（`workflows/`、`input_builders/`、`schedulers/`、`utils/`、`cli/`、`tests/`、
   `docs/`、`AGENT.md`）照常提交。提交前用 `git status` 逐项确认，**不要**用
   `git commit -a` / `git add -A` / `git add .`，只用显式路径 `git add <file>`。

## 3. 目录结构与职责

| 目录 | 职责 | 关键约束 |
|------|------|----------|
| `workflows/<method>/<backend>.py` | 自定义 WorkChain 定义 | 包装插件；输出经 calcfunction |
| `input_builders/<method>/<backend>.py` | `SoftwareAdapter` 子类：preset YAML → AiiDA inputs dict | `_workchain_entry_point()` 返回 `uranium.<method>.<backend>`（或插件入口）；`_build_workchain_inputs` / `_prepare_workflow_inputs` / `adapt` |
| `schedulers/<method>.py` | `WorkflowOrchestrator` 子类 + `register_workflow()` 自注册 | 定义 `ADAPTERS` / `BACKENDS` / `PRESET_SUBKEYS` |
| `schedulers/__init__.py` | import 触发自注册 | 新方法必须在此加 import |
| `parameters/<method>.yml` | 方法级协议（顶层 per-backend 块） | 与 `PRESET_SUBKEYS` 对应；从**数据根**读取 |
| `parameters/<backend>/<category>.yml` | 后端参数（`scf.yml` / `test.yml` / `temporary.yml`） | category 由 input.json 的 dict key 指定；跨方法复用 |
| `static/metadata.yml`、`static/structure.yml` | 站点表（队列/墙钟/结构注册表） | 与 `parameters/` 同属数据根，一起搬家 |
| `utils/` | config / plot / report / elastic / 结构等纯函数 | 无 AiiDA import（config 除外可 lazy） |
| `cli/_common.py` | `METHOD_SPECS` 驱动统一 CLI | 新方法必须加 `MethodSpec` |
| `pyproject.toml` | 打包 + entry points | 与代码引用保持一致 |

## 4. 命名规范

- **backend**：`abacus` / `vasp` / `fleur`（小写，目录名与字符串一致）。
- **method**：`smear` / `convergence` / `magmom` / `banddos` / `relax` / `elastic` /
  `eos` / `phonopy`（单数小写）。
- **entry point**：`uranium.<method>.<backend>`，如 `uranium.elastic.fleur`。
- **参数 key**：`parameters/<method>.yml` 顶层块名 = 后端名；`PRESET_SUBKEYS[backend]`
  指向共享子文件（如 `"scf"` → `parameters/<backend>/scf.yml`）。
- **CLI method 名**：与 method 一致（`--method banddos` 等），与 entry point 解耦。

## 5. 参数体系规范

- **数据根**：`parameters/` 与 `static/` 成对出现在一个「数据根」里，路径运行时解析
  （见 §2.7）。本机数据根 = `aiida-uranium-scripts/`；包里那份是冻结快照。
  **编辑参数 → 改数据根**；`aiida-uranium check -i …` 第一行会打印本次用的数据根与来源。
- 布局：`parameters/<method>.yml` 放方法专属协议；后端 preset 放
  `parameters/<backend>/<category>.yml`（`category` = input.json 里的 dict key），被多方法引用。
- preset 语义：preset 名（`test` / `test_soc` / `pw_r` / `lcao_r` / `nosoc` / `soc` …）
  在文件内注释说明；**新增 preset 必须写注释**（物理含义、适用场景）。
  **同一文件内不要重复定义 preset 名**（YAML 取最后一次，前面的静默失效）。
- 修改参数：先 grep 所有引用方（`input_builders/`、`schedulers/`、测试），确认语义后再改；
  删除 preset 前确认没有调用方依赖。
- 只由顶层控制：input.json 指定的 preset 是唯一来源；不实现"预设不存在就 fallback 到别的
  preset"的逻辑。
- **同步纪律**：数据根里的 preset 与 `static/*.yml` 只留在本地（见 §2.8），不随代码一起
  提交/推送；需要同步时必须由用户当次明确说明。生成/修改参数的脚本、报告可以提交，
  参数本身不提交。

## 6. 如何添加一个新方法（检查清单）

1. `workflows/<method>/<backend>.py`：自定义 WorkChain（包装插件，必要时）；
   输出用 calcfunction 包装。
2. `input_builders/<method>/<backend>.py`：`SoftwareAdapter` 子类，
   `_workchain_entry_point()` 返回正确入口。
3. `schedulers/<method>.py`：`WorkflowOrchestrator` 子类 + `register_workflow(...)`。
4. `schedulers/__init__.py`：加 `from . import <method>`。
5. `parameters/<method>.yml` + 需要的 `parameters/<backend>/<subkey>.yml`。
6. `utils/report/<method>.py`：Markdown 报告生成器。
7. `cli/_common.py`：`METHOD_SPECS` 加 `MethodSpec`；`_common.py` 加
   `<METHOD>_CLASS_TO_BACKEND`。
8. `pyproject.toml`：注册 `uranium.<method>.<backend>` entry points。
9. `tests/`：加单元测试（纯函数）+ scheduler/adapter 测试（mock 不连真实 profile）。
10. README 支持矩阵 + AGENT.md 方法清单同步更新。

## 7. 测试规范

- 运行：`pytest tests/`（全量必须绿）。
- 纯函数（`utils/plot/compare.py`、`utils/elastic.py`、`utils/report/*`）必须有
  **不依赖 AiiDA profile** 的单元测试（默认路径）。
- WorkChain / adapter / scheduler 测试用 mock（`tests/conftest.py` 有现成 fixture），
  不连真实数据库；需要真实 profile 的集成测试单独标注并跳过（CI 无 profile）。
- 数值断言用 `pytest.approx`；NaN 显式处理；不写"恰好相等"的浮点断言。
- 新功能 = 新测试；修 bug = 先加复现测试再改代码。

## 8. 代码风格

- `black`（line-length 88）+ `isort`（profile=black）+ `flake8` + `mypy`（宽松），
  配置在 `pyproject.toml`。提交前跑 `black . && isort .`。
- 所有公开函数/类有 docstring（参数、返回、单位——**物理量必须写单位**，如 eV、GPa、Å³）。
- 类型注解：参数和返回值。

## 9. 变更纪律

- **entry point 改名/增删**：必须同时改 `pyproject.toml` 和代码中所有
  `_workchain_entry_point()` / `WorkflowFactory` 引用 / 测试断言；并提示使用者
  `pip install -e .` + `verdi devel refresh-entry-point-cache`（写入 README/PR 描述）。
- **calcfunction 签名变更**：daemon 中已运行进程不会自动拾取，需
  `verdi daemon restart` 后重新提交受影响的计算。
- **物理/数值语义变更**（如比较指标的公式）：更新 `AGENT.md` 与相关 docstring 中引用的
  文献/定义（如 PRB 98, 085117 的 η_v / max η / ω 定义）。
- README / 本文档与代码同步更新，不允许"文档待补充"长期存在。

## 10. 本机环境提示（当前开发机）

- AiiDA profile：`aiida_profile`（PostgreSQL `aiida_db`，disk-object-store
  `/home/liguozhou/data/aiida_profile/profile`）。
- 数据根：`/home/liguozhou/abacus/calculations/bcc_uranium-v12/aiida-uranium-scripts`
  （含 `parameters/` + `static/`，私有、不进 git）。input.json 放在它下面时自动命中；
  从别处跑就给 `--data-dir` 或 `AIIDA_URANIUM_DATA`。
- editable 安装指向 `bcc_uranium`（v11）那份 clone：在 `-v12` 目录里跑 CLI / pytest 需要
  `PYTHONPATH=src`（或对 v12 做 `pip install -e .`）。
- 沙箱限制由**当前会话**的策略决定（可能是 `workspace-write`，也可能放宽到
  `danger-full-access`）：`workspace-write` 下 `~/.aiida` 与 `/home/liguozhou/data` 只读，
  跑不了依赖 profile 的 CLI / daemon 命令——此时验证数据用 psql 查
  `db_dbnode`/属性 + 读仓库 loose 对象，或把 `AiiDAConfigPathResolver.access_control_dir`
  monkeypatch 到可写 tmp 目录。
- 参考数据：`aiida-uranium-scripts/banddos/nosoc/band.json`（pks：abacus
  `2843fc22`/`89c921e8`、fleur `a91c3e15`）可用于 band-compare 演示。
