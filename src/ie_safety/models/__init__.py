"""建模子包：窗口切分、验证协议、模型、负对照、泄漏审计。"""

from .dataset import WindowSpec, build_window_dataset, make_augmented_dataset  # noqa: F401
from .validation import (  # noqa: F401
    bootstrap_auc_ci,
    hanley_mcneil_se,
    paired_bootstrap_delta,
    repeated_grouped_cv,
    summarize_cv,
)
