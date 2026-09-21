# IE-Alpaca：任务一事故风险预测

项目目录、数据版本和每次实验的评分记录约定见[项目架构与实验迭代](docs/项目架构与实验迭代.md)；模型设计见[任务一事故预测模型设计](任务一事故预测模型设计.md)；每版的真实数据分数与后续决策记入[实验记录](docs/实验记录.md)。正式数据预处理与 V1 CatBoost 训练均已完成；V1 的开发集 20→40 代理 OOF ROC-AUC 为 **0.7073**、Accuracy@0.5 为 **0.7042**，锁定组尚未评分。

每次实验是一个独立模型版本：在数据库目录 `E:\Databases\2026 清华IE亮剑-算法赛道赛题\任务一实验结果\runs\<run_id>` 内同时保存 `source/` 当版代码快照、`predictions/` 预测数据、配置、指标和模型权重；项目仓库保留持续开发的代码。`leaderboard.csv` 汇总每版的开发集 OOF ROC-AUC、Accuracy@0.5、PR-AUC 和 Brier。

当前已完成的首版为 `V1_20260921_112800_71fbec41`。其指标、5 折模型、代码快照和 500 车候选预测都在同名版本目录内。该分数只衡量 06-20 前 20 天预测后 40 天的代理任务，不能当作完整 60→40 任务或官方测试分数。

## 第一版训练入口

第一版需要 `preprocess.py` 已生成的 `daily_features.parquet`、`bag_index.parquet`、`profile.parquet`。这些文件现已保存在数据库的 `任务一预处理结果` 中，并通过 `validation_report.json` 检查。训练命令是：

```powershell
python train_v1.py
```

默认读取数据库的 `任务一预处理结果`，把冻结划分保存在 `任务一实验结果/splits/split_v1_seed2026.json`，并为每次训练新建 `任务一实验结果/runs/V1_.../`。需要指定数据库根目录或处理结果目录时使用 `--database-root`、`--input`、`--results`；这些路径必须位于数据库根目录内。模型参数见 [v1_catboost.json](configs/experiments/v1_catboost.json)，默认 CPU；若本机 CatBoost GPU 可用，可将 `model.device` 设为 `gpu` 并配置 `gpu_devices`。GPU 训练的浮点累加存在非确定性，同一配置重复运行仍保存为不同版本。

`train_v1.py` 仅用 **06-20 前 20 天** 生成历史 near-miss、事件率与行驶暴露量特征，以 06-20 后 40 天代理标签训练。按车辆冻结约 20% 锁定组，余下开发车辆做 5 折 OOF。锁定组**不评分也不参与训练**；每折另训练暴露量回退模型，供锚点时没有事件源的车辆使用。`metrics.json` 的 AUC 和准确率只属于开发集代理任务。候选 500 车预测由开发集模型生成，保存在版本目录的 `predictions/candidate_500.parquet` 和 `predictions/submission_candidate.csv`；后者仅是内部候选，正式提交前还须按组委会模板核对列名和格式。

每版的 `source/` 保存实际代码内容和配置；`models/` 保存折模型与开发集模型；`predictions/oof_predictions.parquet` 可按车复查分数。`leakage_audit.json` 记录车辆划分互斥与时间边界检查，`result_summary.md` 和 `metrics.json` 给出本版结果。程序不会读取 `任务一旧中间结果_未完成`。完整 60 天多预测期网络、LightGBM 与融合仍是下一阶段工作，不属于 V1 的指标。

## 已有预处理入口

`preprocess.py` 把比赛给定的车辆画像、风险事件、轨迹和 IMU 整理为车辆日特征与车辆级 40 天代理标签。原始文件只读；默认按赛题规则排除 2026-07-31 及之后的记录。实现细节和模型边界见[设计文档](任务一事故预测模型设计.md)。

**已完成的正式结果：** `任务一预处理结果` 包含 500 台画像车辆、30,000 行车辆日特征（每车 60 天）、4,000 行样本索引。06-20 的代理标签为 149 正例、329 暂定负例、22 无事件源未标注。轨迹日表覆盖 477 台车，IMU 日表覆盖 478 台车。`model_inputs/` 已生成 478 台可监督车辆的 5 折 20 天张量、85 个日特征及 500 台车的预测包；本机运行使用 CPU。独立检查结果见同目录的 `validation_report.json`，状态为 `passed`。这份张量的 5 折与 V1 CatBoost 另建的开发/锁定组划分不同，不能混作同一组验证分数。

本次实际运行的阶段、质量统计和轨迹旧分片复用依据见[预处理运行记录](docs/预处理运行记录.md)。

## 环境与运行

Python 3.11+，安装依赖：

```powershell
python -m pip install -r requirements.txt
```

完整运行：

```powershell
python preprocess.py --input 'E:\Databases\2026 清华IE亮剑-算法赛道赛题' --stage all --threads 4 --memory-limit 6GB
```

默认输出到 `E:\Databases\2026 清华IE亮剑-算法赛道赛题\任务一预处理结果`。`--output` 如需改名，也必须指向数据库目录内独立的处理结果子目录；不能指向原始 IMU 或轨迹目录。项目目录不写入处理结果。可以按 `profile → events → trajectory → imu → assemble` 顺序分别运行，避免一个阶段失败后重跑前面的大文件。每阶段完成后输出同名 `quality_*.json`。`--hash-files` 会给原文件做 SHA-256，但会额外顺序读取约 57 GB 数据，默认关闭。

轨迹和 IMU 按原始分片处理。中断后用相同输出目录和 `--resume` 复用已经完成的分片，例如：

```powershell
python preprocess.py --input 'E:\Databases\2026 清华IE亮剑-算法赛道赛题' --stage trajectory --resume --threads 4 --memory-limit 6GB
python preprocess.py --input 'E:\Databases\2026 清华IE亮剑-算法赛道赛题' --stage imu --resume --threads 4 --memory-limit 6GB
python preprocess.py --input 'E:\Databases\2026 清华IE亮剑-算法赛道赛题' --stage assemble
```

`--temp-directory` 可指定处理结果子目录中的 DuckDB 临时目录。正常退出时数据库连接会关闭；如果进程被强制终止，输出目录中的 `.partial` 和临时目录可能需要人工清理，已完成的 `.parquet` 分片可以保留。此前项目目录下的 `output` 已完整移到数据库目录 `任务一旧中间结果_未完成` 归档；新默认路径不会自动读取它们。

## 核心输出

| 文件 | 内容 |
|---|---|
| `manifest.json`、`quality_*.json`、`run_config.json` | 来源清单、质量统计和处理参数 |
| `profile.parquet`、`coverage.parquet` | 500 台目标车辆与四类源覆盖情况 |
| `events_clean.parquet`、`accidents.parquet` | 规则时间内的去重事件和事故/未遂事故；速度或坐标无效不删除事故标签 |
| `event_daily.parquet`、`trajectory_daily.parquet`、`imu_daily.parquet` | 各模态按车/日聚合结果 |
| `daily_features.parquet` | 500 台车 × 60 天的日特征与缺测掩码 |
| `bag_index.parquet` | 06-14 至 06-20 的 40 天代理标签、07-30 的待预测索引 |

`bag_index.label_status` 区分 `observed_positive`、`provisional_negative`、`unlabeled_no_event_feed` 和 `target_unknown`。画像与风险事件中未匹配的车辆不被自动标为负例。`daily_features` 中某源缺失用 `NULL` 和覆盖标志表达；真零事件与真零行驶另行保留。

日特征里的 `has_event_feed`、`has_trajectory`、`has_imu` 仅在该车对应数据源**截至当天**已经出现记录时为真；未来才出现的源不会反向填充历史日。`coverage.parquet` 保留全量覆盖信息，供质量检查和标签可用性判断，不进入模型张量。

生成模型可直接读取的张量与车辆级 5 折拆分：

```powershell
python make_model_inputs.py --input 'E:\Databases\2026 清华IE亮剑-算法赛道赛题\任务一预处理结果' --folds 5 --seed 2026 --device auto
```

模型输入默认保存在同一处理结果目录内的 `model_inputs` 子目录。每折有 `fold_N_train.npz`、`fold_N_val.npz` 和 `fold_N_scaler.json`；最终拟合使用 `final_train.npz`、`final_submission.npz`、`final_scaler.json`。数组包含 `x`（样本×20天×特征）、`value_mask`、`day_mask`、`static_energy`、设备号、锚点日与标签。缺失值在保留掩码后填 0；`-1` 标签仅代表待预测或无事件源，不能用于监督。分位截尾、`log1p`、中位数/IQR 标准化只拟合各折训练车辆 **06-01 至 06-14** 的历史日，避免用代理锚点之后的信息。`splits.json` 保存各折车辆名单、特征列顺序和实际使用的计算设备。

重新核查处理结果时运行：

```powershell
python validate_preprocessing.py --input 'E:\Databases\2026 清华IE亮剑-算法赛道赛题\任务一预处理结果'
```

该检查核对车日与标签行数、时间边界、事故标签与事故表、张量形状和各折车辆互斥，并在处理结果目录写入 `validation_report.json`；不训练模型。

`--device auto` 在 CUDA 版 PyTorch 可用时用 GPU 做张量截尾、`log1p` 与标准化，否则用 NumPy CPU；`--device cuda` 要求 GPU 可用，不满足时明确报错；`--device cpu` 不需要 PyTorch。PyTorch 的 CUDA 安装方式取决于显卡驱动和运行环境，请按其官方安装说明选择匹配版本。DuckDB 的大文件读取、排序和聚合仍在 CPU 上分片完成；最终 `.npz` 文件保存在磁盘，训练代码之后可再将批次送到 GPU。当前车辆日特征规模较小，CUDA 不保证比 CPU 更快。

当前 IMU 分支先输出**旋转不变**的加速度/角速度模长窗口摘要。设备姿态与陀螺仪单位未经全量验证，因此尚不输出纵向、横向、垂直轴的驾驶风险特征。`trajectory_daily` 的增量里程、行驶时长、GPS 位移核查量和异常计数同时保留，供质量复核；GPS 位移核查量不进入模型张量。近半年画像统计的计算截止日不明，预处理会保留原值，但训练时应按设计文档暂不使用这些可能泄漏的列。

## 检查

```powershell
python -m unittest discover -s tests -v
```

测试覆盖 40 天标签边界、07-31 排除、缺失事件源、IMU `\N`、日历行数、四源端到端处理，以及 V1 的时间泄漏、冻结划分和训练产物留档。`make_model_inputs.py` 是既有 20 天张量生成器；V1 表格模型直接读取已处理的 Parquet 文件，不读取其 `.npz` 输出。
