# V12 代码运行说明

## 1. 环境

推荐 Python 3.11+。在本目录执行：

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
```

## 2. 从原始比赛数据预处理

```powershell
python preprocess.py --input '<比赛数据库根目录>' --stage all --threads 4 --memory-limit 6GB
python validate_preprocessing.py --input '<比赛数据库根目录>\任务一预处理结果'
```

## 3. 使用随包模型直接复现两项结果

```powershell
python generate_submission_outputs_v12.py `
  --input '<比赛数据库根目录>\任务一预处理结果' `
  --checkpoint 'artifacts\task1_run\models\development_state_mae.pt' `
  --splits 'artifacts\task1_run\splits.json' `
  --task2-config 'configs\experiments\task2_v12_scorecard.json' `
  --output 'reproduced_outputs' `
  --device auto
```

生成的`task1_predictions.csv`和`task2_safety_scores.csv`应分别与提交包中的正式结果一致。`--device auto`优先使用CUDA，没有CUDA时回退CPU。

## 4. 重新训练V12

```powershell
python train_v12.py --config configs\experiments\v12_risk_chain_ssl.json
```

该命令完成车辆级五折实验和模型消融。若要重新执行最终重训，将打印出的父运行ID填入`configs/experiments/v12_finalize_state_mae.json`的`parent_run_id`，再运行：

```powershell
python finalize_v12.py --config configs\experiments\v12_finalize_state_mae.json
```

## 5. 关键入口

- `preprocess.py`：四类原始数据清洗与日级汇总；
- `train_v12.py`：State-MAE、动态预测和风险链消融；
- `finalize_v12.py`：按停止规则重训最终State-MAE；
- `generate_submission_outputs_v12.py`：从所附权重直接生成两个任务结果；
- `score_task2_v12.py`：任务二版本化评分流水线；
- `validate_task2_outputs.py`：任务二交付校验。

所有训练与评分仅使用赛方提供数据，没有使用外部数据。
