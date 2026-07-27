# aiida-uranium-workflow

AiiDA 工作流模板仓库，用于 BCC 铀的 smear/sigma 参数扫描与电子熵对比研究。

## SmearSigmaWorkflow

定义两个工作流，一个是 abacus 工作流，一个是 vasp 工作流。完成以下的功能：

工作流一共有以下的输入参数：
1. workchain: 指定workchain的名称，abacus或vasp和该workflow的名称。

一共有三部分输入：
1. cmd下指定的输入参数：包含workchain、profile、code。
2. 项目中`protocol/vasp/smear_sigma.json`包含parameters、structure、kpoints_spacing、potential_family、potential_mapping。`protocol/abacus/smear_sigma.json`包含parameters、structure、kpoints_distance、pseudo_family。
3. 默认参数（smear / sigma 扫描列表）。
4. 调用workchain可以生成一个Report文件，包含计算结果，包括电子熵等。

## 待补充

- 两个 workchain 的具体实现
- Report 文件生成逻辑
- 测试用例
