# IE-Alpaca：任务一事故风险预测

## 当前进展：V9 人类先验、风险链与暴露分解

V9 已按[预注册设计](docs/V9因果先验与风险链模型设计.md)完成三个阶段。E0 在完全相同的 V3 线性风险模型上逐步加入事件角色汇总、同日共现和方向性风险链；方向链版 20→40 OOF AUC 为 **0.7717**，低于未增加先验特征的 V3 复现值 **0.7875**。E1 的分层加性网络、长期/近期拆分和固定交互最高 AUC **0.7249**。E4 的直接风险、暴露辅助和暴露因子化模型分别为 **0.7504 / 0.7490 / 0.7520**，相邻改动的配对区间均跨 0。

预先规定只有 E1 有效才训练质量门控 E2、只有方向链有效才训练 ChainNet E3；两个前提都没有成立，因此 E2/E3 已按停止规则终止。V3 Model 0 继续作为当前主模型，96 台锁定车仍未评分。

```powershell
python train_v9_e0.py
python train_v9_e1.py
python train_v9_e4.py
```

三个已完成运行分别是 `V9E0_20260923_105308_2dc5cf83`、`V9E1_20260923_125713_dbff0a36` 和 `V9E4_20260923_132215_dbfd0d62`；源码快照、模型、OOF、配对区间及每个变体的 500 车候选均保存在数据库运行目录。

## 当前进展：V8 深层非 Transformer 对照

已在V5基础上完成两个不使用Transformer的深度实验。V8-A把事件支路改为`9→16→8→1`、上下文支路改为`34→32→16→1`，保留加性融合；V8-B进一步把5个风险事件族、2个质量信号和16维上下文表示送入联合融合头。20→40开发OOF AUC分别为 **0.7139** 和 **0.7223**，均未超过V5的 **0.7307**，也未超过V3的 **0.7875**。完整方法和区间见[V8深层非Transformer实验](docs/V8深层非Transformer实验.md)。

```powershell
python train_v8.py --config configs/experiments/v8a_deep_additive.json
python train_v8.py --config configs/experiments/v8b_deep_fusion.json
```

运行目录分别为`V8A_20260923_062423_e43517e4`和`V8B_20260923_062529_f7cb3880`。两份三列交付候选已经导出，96台锁定车仍未评分。

## 当前进展：V7 轻量 Transformer

已实现并训练 [V7 轻量 Transformer 风险模型](docs/V7轻量Transformer风险模型设计.md)：以 24 类事件及暴露、轨迹和 IMU 的逐日记录构造最长 60 天序列，用一层小型 Transformer 检验行为顺序是否提供汇总特征以外的信息。风险目标、右删失损失、冻结车辆划分和交付格式继续沿用 V3–V6。20→40 开发车 OOF AUC **0.7569**、Brier **0.1832**、Accuracy@0.5 **0.7094**；同输入的无注意力对照 AUC **0.7583**。注意力本身未表现出增益，V3 的 **0.7875** 仍为开发集最优。96 台锁定车未评分。

```powershell
python train_v7.py --config configs/experiments/v7_temporal_transformer.json
```

已完成运行 `V7_20260922_154255_182bd4e7` 保存在数据库的 `任务一实验结果/runs/`；三列交付表为该运行下的 `deliverables/task1_v7_predictions.csv`。每次运行生成新文件夹，不覆盖这一版。

## 当前进展：V6 参数量实验

V6 保持 V5 的特征、风险损失与按车选轮，只将事件网络和非事件网络的隐藏层分别扩至 `16/32`（1,360 参数）或 `32/64`（2,688 参数）。两档的 20→40 开发 OOF AUC 分别为 **0.7155** 和 **0.7186**，均未超过 V5 的 **0.7307**。完整协议、配对区间和选轮结果见 [V6 参数量实验](docs/V6参数量实验.md)。运行入口：

```powershell
python train_v6.py --config configs/experiments/v6_width_medium.json
python train_v6.py --config configs/experiments/v6_width_large.json
```

每档独立保存源码、权重、逐车 OOF、选轮曲线及 500 车候选。中档参考运行是 `V6_20260922_104741_e3b64527`，大档为 `V6_20260922_104858_effdbc14`；各自目录内有三列 `deliverables/task1_v6_predictions.csv`。本轮不替换 V5，也不替换当前开发集表现更好的 V3 Model 0。96 台锁定车仍未评分。

## 上一版：V5 内层选轮的全事件 PyTorch 风险模型

V5 保持 V4 的全部特征、网络和右删失损失，仅将固定 30 轮改成**按车辆隔离的内层验证选轮**。每个外折只在训练车中选择轮数，随后在该折全部训练车上重训；96 台锁定车未参与。运行入口和完整协议见 [V5 训练轮数实验](docs/V5训练轮数实验.md)：

```powershell
python train_v5.py
```

默认上限 100 轮的运行 `V5_20260922_103523_4d54176d`，20→40 开发 OOF AUC **0.7307**，Brier **0.1860**，Accuracy@0.5 **0.7173**；V4 固定 30 轮 AUC **0.7009**。另将上限延至 200 轮做敏感性检查，AUC **0.7296**，没有可辨认的增益。两个 V5 运行都保存了源码、选轮曲线、折模型、OOF 和 500 车候选；三列交付候选分别位于本版目录的 `deliverables/task1_v5_predictions.csv`。V5 仍低于 V3 Model 0 的 **0.7875**，故当前不替换 V3。完整 60→未来40 天没有本地真值。

## 上一版：V4 全事件 PyTorch 风险模型

V4 已实现并运行：读取现成的 500 车 × 60 日表，对全部 24 种行为和设备事件分别保留报警段数、原始次数、近期变化和最近发生时间；PyTorch 小网络学习每种事件的相对权重，输出每日事故/未遂事故风险率，再换算未来 40 天概率。设计和边界见 [V4 全事件权重风险模型](docs/V4全事件权重风险模型.md)。运行入口与配置：

```powershell
python train_v4.py
```

训练会沿用冻结的 382 台开发车、5 折和 96 台锁定车，只在开发车计算 OOF。修正后的 V4 运行 `V4_20260922_093849_3132d976` 在 20→40 代理任务上的 OOF AUC 为 **0.7009**，Accuracy@0.5 为 **0.7094**，Brier 为 **0.1926**。统一事件权重对照 AUC 为 **0.6987**；V3 Model 0 为 **0.7875**，因此 V4 暂不替代 V3。前一轮只用报警段数的 V4 原型 `V4_20260922_093038_26842a0e` 也保留在数据库中，AUC 为 **0.6889**。两轮均未评分 96 台锁定车。运行使用 CPU；配置中的 `model.device` 可设为 `auto`、`cpu` 或 `cuda`。

每轮源码快照、PyTorch 权重、各事件权重、逐车 OOF 和 500 车候选预测位于数据库的 `任务一实验结果/runs/<run_id>/`。V4 的三列交付候选在修正版运行的 `deliverables/task1_v4_predictions.csv`，列名和 V3 相同：`gpsno,prediction,probability`。重新导出可运行：

```powershell
python export_task1_v4.py --run-dir 'E:\Databases\2026 清华IE亮剑-算法赛道赛题\任务一实验结果\runs\V4_20260922_093849_3132d976' --output 'E:\Databases\2026 清华IE亮剑-算法赛道赛题\任务一实验结果\runs\V4_20260922_093849_3132d976\deliverables\task1_v4_predictions_copy.csv'
```

导出脚本要求目标文件不存在，以防覆盖历史交付。Day60→未来40 天没有本地真实标签，V4 文件仍是候选结果。完整实验对比见[实验记录](docs/实验记录.md)。

## 既有版本：V3 Model 0、V2 与 V1

**V3 Model 0 已完成：** [V3 Landmark 风险模型](任务一事故预测模型设计.md)以车辆级历史窗口和右删失 next-event hazard 为主。首个 elastic-net 逻辑回归版本已训练；浅层 CatBoost 留作下一轮对照。V1/V2 的已有结果和运行方式保留在下文。

运行 V3 Model 0：

```powershell
python train_v3_model0.py
```

默认读取数据库的 `任务一预处理结果`，在数据库 `任务一实验结果/runs/V3M0_.../` 新建版本文件夹，保存当版代码、模型、六个锚点的开发集 OOF、指标和 500 车 Day60 候选预测。配置见 [v3_model0_logistic.json](configs/experiments/v3_model0_logistic.json)。该线性模型使用 CPU；96 台锁定测试车尚未评分。22 台事件表无匹配记录且无轨迹/IMU 记录的车使用开发车先验，输出标记为低证据。

**任务一表格交付：** 已将 V3 Model 0 的 500 车 Day60 候选预测导出为数据库版本目录内的 `deliverables/task1_v3_model0_predictions.csv`。列为 `gpsno,prediction,probability`：`gpsno` 对应原始车辆画像的设备号，`prediction` 是固定 0.5 阈值得到的 0/1，`probability` 是未来 40 天至少一次事故或未遂事故的模型概率。赛题说明只规定 CSV 内容，没有规定列名；若平台另给模板，应按其列名重排。复现命令：

```powershell
python export_task1_v3.py --run-dir 'E:\Databases\2026 清华IE亮剑-算法赛道赛题\任务一实验结果\runs\V3M0_20260921_155138_475c1ea1'
```

导出脚本检查 500 个设备号与原始画像对应、概率范围、二分类阈值、Day60→40 边界和 22 台低证据车辆。CSV 只含交付要求的三列，不含真实标签、验证折或内部路由。任务一另需算法说明 PDF 和可复现代码，见赛题说明第 6 节；本 CSV 的 60→40 结果没有本地真值可评分。

`train_v2.py` 读取数据库内已生成的 500 车×60 日特征，按冻结的车辆名单做 5 折神经网络训练；训练车、验证车和 96 台锁定车整车隔离。每折在训练车完整 60 天上做遮蔽重建，并用可完整观测的 7/14/21/30/40 天结局训练风险头。07-30 的完整 60 天历史用于生成 500 车未来 40 天候选概率。

```powershell
python -m pip install -r requirements.txt
python train_v2.py
```

配置见 [v2_temporal.json](configs/experiments/v2_temporal.json)，分数和逐车预测保存在数据库 `任务一实验结果/runs/V2_.../`，同目录还保存本版代码快照、折模型、标准化器和训练日志。V2 复用 V1 的 `split_v1_seed2026.json`；不要使用旧 `model_inputs/` 的另一套 5 折划分。完整协议、可观察标签的边界和目录说明见[项目架构与实验迭代](docs/项目架构与实验迭代.md)。

**分数口径：** 开发集主指标是 06-20 的 **20 天历史→后 40 天**代理 OOF；60 天历史→未来 40 天没有给定真值，候选结果没有本地实测 AUC 或准确率。锁定车的已知代理标签本版不评分。

项目目录、数据版本和每次实验的评分记录约定见[项目架构与实验迭代](docs/项目架构与实验迭代.md)；V3 模型设计见[任务一事故预测模型设计](任务一事故预测模型设计.md)；每版的真实数据分数与后续决策记入[实验记录](docs/实验记录.md)。同一车辆划分的开发集 20→40 OOF ROC-AUC：V1 **0.7073**，V2 **0.6946**，V3 Model 0 **0.7875**，V4 **0.7009**，V5 上限100 **0.7307**，V6中档 **0.7155**、大档 **0.7186**，V7 Transformer **0.7569**，V8-A **0.7139**、V8-B **0.7223**，V9方向链 **0.7717**、层级加性 **0.7249**、暴露因子化 **0.7520**；96台锁定车尚未评分。

每次实验是一个独立模型版本：在数据库目录 `E:\Databases\2026 清华IE亮剑-算法赛道赛题\任务一实验结果\runs\<run_id>` 内同时保存 `source/` 当版代码快照、`predictions/` 预测数据、配置、指标和模型权重；项目仓库保留持续开发的代码。`leaderboard.csv` 汇总每版的开发集 OOF ROC-AUC、Accuracy@0.5、PR-AUC 和 Brier。

已完成版本为 `V1_20260921_112800_71fbec41` 和 `V2_20260921_135210_2fac9c14`。各版指标、5 折模型、代码快照和 500 车候选预测都在同名版本目录内。开发分数只衡量 06-20 前 20 天预测后 40 天的代理任务，不能当作完整 60→40 任务或官方测试分数。

## 第一版训练入口

第一版需要 `preprocess.py` 已生成的 `daily_features.parquet`、`bag_index.parquet`、`profile.parquet`。这些文件现已保存在数据库的 `任务一预处理结果` 中，并通过 `validation_report.json` 检查。训练命令是：

```powershell
python train_v1.py
```

默认读取数据库的 `任务一预处理结果`，把冻结划分保存在 `任务一实验结果/splits/split_v1_seed2026.json`，并为每次训练新建 `任务一实验结果/runs/V1_.../`。需要指定数据库根目录或处理结果目录时使用 `--database-root`、`--input`、`--results`；这些路径必须位于数据库根目录内。模型参数见 [v1_catboost.json](configs/experiments/v1_catboost.json)，默认 CPU；若本机 CatBoost GPU 可用，可将 `model.device` 设为 `gpu` 并配置 `gpu_devices`。GPU 训练的浮点累加存在非确定性，同一配置重复运行仍保存为不同版本。

`train_v1.py` 仅用 **06-20 前 20 天** 生成历史 near-miss、事件率与行驶暴露量特征，以 06-20 后 40 天代理标签训练。按车辆冻结约 20% 锁定组，余下开发车辆做 5 折 OOF。锁定组**不评分也不参与训练**；每折另训练暴露量回退模型，供锚点时没有事件源的车辆使用。`metrics.json` 的 AUC 和准确率只属于开发集代理任务。候选 500 车预测由开发集模型生成，保存在版本目录的 `predictions/candidate_500.parquet` 和 `predictions/submission_candidate.csv`；后者仅是内部候选，正式提交前还须按组委会模板核对列名和格式。

每版的 `source/` 保存实际代码内容和配置；`models/` 保存折模型与开发集模型；`predictions/oof_predictions.parquet` 可按车复查分数。`leakage_audit.json` 记录车辆划分互斥与时间边界检查，`result_summary.md` 和 `metrics.json` 给出本版结果。程序不会读取 `任务一旧中间结果_未完成`。本节只描述 V1，V2 多预测期网络见本页开头；LightGBM 与融合仍待开发。

## 已有预处理入口

`preprocess.py` 把比赛给定的车辆画像、风险事件、轨迹和 IMU 整理为车辆日特征与车辆级 40 天代理标签。原始文件只读；默认按赛题规则排除 2026-07-31 及之后的记录。实现细节和模型边界见[设计文档](任务一事故预测模型设计.md)。

**已完成的预处理结果：** `任务一预处理结果` 包含 500 台画像车辆、30,000 行车辆日特征（每车 60 天）、4,000 行样本索引。06-20 的代理标签为 149 正例、329 暂定负例；另有 22 台事件表无可匹配记录，未用于监督。轨迹日表覆盖 477 台车，IMU 日表覆盖 478 台车。`model_inputs/` 已生成 478 台候选监督车辆的 5 折 20 天张量、85 个日特征及 500 台车的预测包；本机运行使用 CPU。独立检查结果见同目录的 `validation_report.json`，状态为 `passed`。这份张量的 5 折与 V1/V3 使用的冻结开发/锁定组划分不同，不能混作同一组验证分数。

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

模型输入默认保存在同一处理结果目录内的 `model_inputs` 子目录。每折有 `fold_N_train.npz`、`fold_N_val.npz` 和 `fold_N_scaler.json`；最终拟合使用 `final_train.npz`、`final_submission.npz`、`final_scaler.json`。数组包含 `x`（样本×20天×特征）、`value_mask`、`day_mask`、`static_energy`、设备号、锚点日与标签。缺失值在保留掩码后填 0；`-1` 标签代表待预测或事件表无可匹配记录，不能用于监督。分位截尾、`log1p`、中位数/IQR 标准化只拟合各折训练车辆 **06-01 至 06-14** 的历史日，避免用代理锚点之后的信息。`splits.json` 保存各折车辆名单、特征列顺序和实际使用的计算设备。

重新核查处理结果时运行：

```powershell
python validate_preprocessing.py --input 'E:\Databases\2026 清华IE亮剑-算法赛道赛题\任务一预处理结果'
```

该检查核对车日与标签行数、时间边界、事故标签与事故表、张量形状和各折车辆互斥，并在处理结果目录写入 `validation_report.json`；不训练模型。

`--device auto` 在 CUDA 版 PyTorch 可用时用 GPU 做张量截尾、`log1p` 与标准化，否则用 NumPy CPU；`--device cuda` 要求 GPU 可用，不满足时明确报错；`--device cpu` 不需要 PyTorch。PyTorch 的 CUDA 安装方式取决于显卡驱动和运行环境，请按其官方安装说明选择匹配版本。DuckDB 的大文件读取、排序和聚合仍在 CPU 上分片完成；最终 `.npz` 文件保存在磁盘，训练代码之后可再将批次送到 GPU。当前车辆日特征规模较小，CUDA 不保证比 CPU 更快。

当前 IMU 分支先输出**旋转不变**的加速度/角速度模长窗口摘要。设备姿态与陀螺仪单位未经全量验证，因此尚不输出纵向、横向、垂直轴的驾驶风险特征。`trajectory_daily` 的增量里程、行驶时长、GPS 位移核查量和异常计数同时保留，供质量复核；GPS 位移核查量不进入模型张量。近半年画像统计的计算截止日不明，预处理会保留原值，但训练时应按设计文档暂不使用这些可能泄漏的列。

## 检查

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
```

测试覆盖 40 天标签边界、07-31 排除、缺失事件源、IMU `\N`、日历行数、四源端到端处理，以及 V1 的时间泄漏、冻结划分和训练产物留档。`make_model_inputs.py` 是既有 20 天张量生成器；V1 表格模型直接读取已处理的 Parquet 文件，不读取其 `.npz` 输出。
