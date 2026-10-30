# VkFFT tuning experiments

实验源码、协议、运行记录和审计工具。上游 VkFFT 以子模块固定版本，不向上游仓库提交实验修改。

```text
git clone --recurse-submodules https://github.com/r7oromen-hub/vkfft__tuning.git
```

最新研究实现位于 `test/plan_selection_study/expanded_search/`。实验版本规则见 [experiments/README.md](experiments/README.md)，历史证据索引见 `experiments/history/`。

历史 GPU 测试发生在本仓库建立之前；归档 commit 不等于当时执行 commit。原始大文件保留在本地结果目录，本仓库中的路径和 SHA-256 只用于定位及校验，不代表原始数据已经上传到 GitHub。

未来测试通过 `tools/run_experiment.py` 创建独立运行声明与结果提交。此仓库导入不代表重新运行了全部历史 CPU/GPU 测试。
