# V12 最终提交说明

本文件夹包含任务一事故风险预测和任务二司机安全评价的正式结果、Markdown模型说明以及复现代码。

## 正式结果

- `01_任务一预测结果/task1_predictions.csv`：500辆车的二分类结果和风险概率。
- `02_任务二安全评价/task2_safety_scores.csv`：500辆车的0–100分安全分。
- `02_任务二安全评价/任务二司机安全评价模型说明.md`：评分维度、权重、公式、V12分层风险链和车队运营建议。
- `02_任务二安全评价/支持材料/task2_driver_scorecards.csv`：逐车六项扣分、等级、主要风险和管理建议。
- `02_任务二安全评价/支持材料/task2_event_weights.csv`：各维度内具体事件的学习权重。

## 文档

任务一和任务二说明均使用Markdown格式。本提交包不包含PDF。

## 复现

先完成数据预处理和V12任务一训练，再运行：

```powershell
python score_task2_simple_v12.py --config configs/experiments/task2_simple_score_v12.json
```

任务二评分采用六个维度。六项扣分之和能够逐项还原安全总分。证据不足车辆标记为U级，补齐设备和数据证据前不用于奖惩。
