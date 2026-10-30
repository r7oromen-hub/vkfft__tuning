# 实验版本与证据管理

本仓库管理实验源码、配置和运行索引。`VkFFT` 是固定到上游 commit 的子模块；克隆使用 `git clone --recurse-submodules`。

## 每次新实验

1. 先提交全部代码、配置和依赖变化，保证主仓库及子模块干净。凭据只通过环境或系统凭据存储提供，不写入代码或命令行记录。
2. 使用 `tools/run_experiment.py` 启动一次有唯一 ID 的实验。它先提交运行声明，得到实际执行 commit，再执行命令；完成或失败后另提交运行记录。
3. 一次实验可包含多个候选/随机种子/重复测量。它们共享同一个执行 commit，并应在原始日志中保留各自 ID。不为每个相同代码的 kernel 调用创建空提交。
4. 大型输出放入 `experiments/artifacts/<id>/`（不进 Git）；运行器对文件记录 SHA-256、长度和路径。结果记录、必要的小型汇总表另行提交。只有本地路径和哈希不等于云备份，原始数据需单独备份。
5. 运行后 `git push origin main`。运行器不会自动上传，避免长任务失败被上传故障覆盖。

示例（在仓库根目录执行；先将 config.json 提交到仓库）：

```text
python tools/run_experiment.py --id example-001 --protocol experiments/config.json -- python path/to/test.py
```

运行器给子进程提供 `EXPERIMENT_ID`、`EXPERIMENT_CODE_COMMIT` 和 `EXPERIMENT_ARTIFACT_DIR`。远程任务应在远端检出该 commit，核对干净状态，并把远端环境、GPU UUID、编译器、驱动、启动命令及二进制/生成头哈希保存到输出目录。仅从本地调用 SSH 不能证明远端运行了同一源码；运行记录会明确保留这个验证责任。

若进程或机器突然退出，运行声明 commit 仍存在，但结果可能缺失；不得标成成功。不要覆写已有实验 ID。失败重试使用新 ID。

## 历史实验

`experiments/history/` 是事后归档索引：每个历史阶段独立提交，记录日志和冻结源码的哈希、现有上游版本以及证据边界。`execution_commit` 为 null 表示当时没有实验仓库提交，不能把今天的归档提交假称为当时运行版本。

首次代码导入只是当前本地快照。某些历史冻结脚本可能与当前文件不同，以历史 manifest 和归档为准。旧日志、密码、二进制和 GB 级归档不会自动提交到 GitHub。
