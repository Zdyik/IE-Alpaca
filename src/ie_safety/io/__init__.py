"""数据读取子包：负责文件发现、列名归一、类型规整与分块读取。"""

from .readers import (  # noqa: F401
    Source,
    discover_datasets,
    iter_events,
    load_events,
    load_profile,
    naive_ts,
    parse_datetime,
    parse_ratio,
    to_gpsno_str,
    to_naive_local_date,
)
