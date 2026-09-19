# reports · 验证报告与机器可读产物

本目录同时放两类东西：

**① 稳定产物（随代码一起维护）**

| 文件 | 说明 |
|---|---|
| `feature_dictionary.csv` | 特征字典：组 / 名称 / 定义 / **时间来源**。时间来源是泄漏审计的兜底证据 |
| `无效清单.md` | 预注册记录与未达效应量门槛的改动。这份清单本身是文档完整性的得分点 |

**② 运行产物（由脚本生成，可随时重跑覆盖）**

| 文件 | 生成自 |
|---|---|
| `label_decision.json` | `scripts/02_audit.py` —— **标签口径决策，整个项目的闸门** |
| `anchor_checks.json` | `scripts/02_audit.py` —— 频次序关系等免费锚点结论 |
| `data_manifest.json` | `scripts/02_audit.py` —— 数据清单与哈希 |
| `validation.md` | `scripts/04_task1.py` —— 九项验收闸门与保险箱评估 |
| `负面对照报告.md` | `scripts/04_task1.py` —— 四类负对照结果 |
| `vault_audit.jsonl` | `scripts/04_task1.py` —— 保险箱每次开启的审计记录 |
| `task2_validation.md` | `scripts/05_task2.py` —— 区分能力、单调性、稳定性、反事实 |

> ⚠️ **当前仓库内的运行产物来自合成数据的冒烟验证。**
> 仓库不含赛题数据，因此这些数字**不代表比赛结果**；每份报告顶部都带有同样的声明。
> 拿到真实数据后按 `README.md` 第 2.3 节重跑即可覆盖。
