对cli的测试：

对src/aiida_uranium_workflow/example中的json进行测试。
run：
    base.json：
    1. 测试是否能从`"workflow": "base"`中调用"abacus"的WorkflowFactory("abacus.base")和"vasp"的WorkflowFactory("vasp.v2.vasp")
    2. 测试