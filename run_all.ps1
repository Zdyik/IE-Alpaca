# 一键复现脚本
#
# 用法：
#   pwsh -File run_all.ps1                 # 完整流程（含负对照，较慢）
#   pwsh -File run_all.ps1 -Quick          # 跳过耗时项（冒烟验证用）
#   pwsh -File run_all.ps1 -Synthetic      # 用合成数据跑（无真实数据时）
#
# 说明：每一步都打印耗时；任一步失败即中止并给出退出码，避免「部分成功」被误当成功。

param(
    [switch]$Quick,
    [switch]$Synthetic,
    [string]$Raw = "",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# ---- 选择解释器：优先 .venv，其次系统 python ----
if (-not $Python) {
    if (Test-Path ".venv\Scripts\python.exe") { $Python = ".venv\Scripts\python.exe" }
    elseif (Test-Path ".venv/bin/python")     { $Python = ".venv/bin/python" }
    else                                      { $Python = "python" }
}
Write-Host "使用解释器: $Python" -ForegroundColor Cyan

# 让 scripts/ 下的脚本能 import ie_safety
$env:PYTHONPATH = "$Root\src"
if (-not $env:MPLCONFIGDIR) { $env:MPLCONFIGDIR = "$Root\.tmp\mpl" }

$failures = @()

function Invoke-Step {
    param([string]$Name, [string[]]$Args)
    Write-Host "`n=== $Name ===" -ForegroundColor Yellow
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Python @Args
    $code = $LASTEXITCODE
    $sw.Stop()
    if ($code -ne 0) {
        Write-Host "  ✗ 失败（退出码 $code），耗时 $([math]::Round($sw.Elapsed.TotalSeconds,1))s" -ForegroundColor Red
        $script:failures += $Name
    } else {
        Write-Host "  ✓ 完成，耗时 $([math]::Round($sw.Elapsed.TotalSeconds,1))s" -ForegroundColor Green
    }
    return $code
}

# ---- 0) 数据准备 ----
if ($Synthetic) {
    Invoke-Step "生成合成数据" @("scripts/01b_make_synthetic.py", "--out", "data/raw/synthetic", "--vehicles", "120") | Out-Null
    $Raw = "data/raw/synthetic"
}

if (-not (Test-Path "data/raw") -or -not (Get-ChildItem "data/raw" -Recurse -File -ErrorAction SilentlyContinue)) {
    Write-Host @"

未在 data/raw 下发现任何数据文件。

两种做法：
  1) 把赛题提供的四个数据集（车辆画像 / 风险事件 / IMU / 轨迹）放入 data/raw/；
  2) 用合成数据先验证链路：  pwsh -File run_all.ps1 -Synthetic

"@ -ForegroundColor Yellow
    exit 2
}

$rawArg = @()
if ($Raw) { $rawArg = @("--raw", $Raw) }

# ---- 主流程 ----
Invoke-Step "02 数据审计与标签闸门" (@("scripts/02_audit.py") + $rawArg) | Out-Null
Invoke-Step "03 特征工程 v1"        (@("scripts/03_features.py") + $rawArg) | Out-Null

$t1args = @("scripts/04_task1.py") + $rawArg
if ($Quick) { $t1args += @("--skip-negative-controls", "--n-repeats", "2") }
Invoke-Step "04 任务一建模与验收" $t1args | Out-Null

Invoke-Step "05 任务二评分模型" (@("scripts/05_task2.py") + $rawArg) | Out-Null
Invoke-Step "06 导出文档 PDF"   @("scripts/06_docs.py") | Out-Null

# ---- 自检与合规（顺序刻意放在最后）----
$vargs = @("scripts/99_verify.py")
if ($Quick) { $vargs += @("--quick") }
Invoke-Step "99 端到端自检" $vargs | Out-Null
Invoke-Step "98 入库合规校验" @("scripts/98_compliance_check.py") | Out-Null

# ---- 汇总 ----
Write-Host "`n$('=' * 62)" -ForegroundColor Cyan
if ($failures.Count -eq 0) {
    Write-Host "全部步骤通过。提交物位于 outputs/，报告位于 reports/，文档位于 docs/。" -ForegroundColor Green
    Write-Host "提醒：outputs/ 与 data/ 下的文件含车辆级信息，不入仓库，只走官方渠道提交。" -ForegroundColor Yellow
    exit 0
} else {
    Write-Host "以下步骤失败：$($failures -join ', ')" -ForegroundColor Red
    exit 1
}
