"""00 · 腾讯云 COS 分享链接探测与解密（数据获取前置步骤）。

## 背景

赛题数据通过四个腾讯云 COS **分享链接**分发，页面是 Vue SPA，直接 HTTP 取不到
文件列表。本脚本把前端接口与加密方案复现出来，用于：

1. 核对四个链接指向的 Bucket / Prefix / Region 是否与预期一致；
2. 在有签名方案时脚本化下载；否则回退到浏览器手动下载。

## 已探明的接口与加密（本脚本即为其可执行版本）

1. ``POST https://cosbrowser.cloud.tencent.com/api/share/query``
   body ``{"id": "<链接里的 id>", "extract": "<提取码>"}`` → 返回 ``{token, accessCode}``
2. 解密：**AES-256-CBC，key = UTF-8(accessCode)，IV = UTF-8("cosbrowser-share")，
   PKCS7 填充，密文 = base64(token)** → 得到一段 query string，
   含 ``Bucket / Prefix / Region / action / expire``

> ⚠️ **不要把链接、提取码或解密结果写进任何入库文件。** 赛题明文禁止泄露数据
> 相关信息，而本仓库是公开的。因此本脚本只从命令行/环境变量接收参数，不内置
> 任何链接，并且输出默认只打印摘要（隐藏 token）。

用法::

    python scripts/00_probe_share.py --id <分享id> --extract <提取码>
    python scripts/00_probe_share.py --from-pdf 赛题说明.pdf   # 仅演示，需自行提供 PDF
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.parse
import urllib.request

API = "https://cosbrowser.cloud.tencent.com/api/share/query"
IV = b"cosbrowser-share"


def decrypt_config(token_b64: str, access_code: str) -> str:
    """复现前端 ``ee(e, t)``：AES-256-CBC(key=accessCode, iv='cosbrowser-share')。"""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    ct = base64.b64decode(token_b64)
    dec = Cipher(algorithms.AES(access_code.encode()), modes.CBC(IV)).decryptor()
    pt = dec.update(ct) + dec.finalize()
    return pt[: -pt[-1]].decode("utf-8", "replace")


def query_share(share_id: str, extract: str, timeout: int = 25) -> dict:
    req = urllib.request.Request(
        API,
        data=json.dumps({"id": share_id, "extract": extract}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="探测腾讯云 COS 分享链接")
    ap.add_argument("--id", required=True, help="分享链接里的 id 参数")
    ap.add_argument("--extract", required=True, help="提取码")
    ap.add_argument("--show-token", action="store_true", help="打印解密出的原始串（默认隐藏）")
    args = ap.parse_args(argv)

    try:
        data = query_share(args.id, args.extract)
    except Exception as exc:
        print(f"请求失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        print("若为网络问题，请改用浏览器手动下载（见 README「数据获取」一节）。", file=sys.stderr)
        return 1

    if data.get("code") != 0:
        print(f"接口返回错误：code={data.get('code')} message={data.get('message')}", file=sys.stderr)
        return 1

    payload = data["data"]
    raw = decrypt_config(payload["token"], payload["accessCode"])
    cfg = {k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}

    print("解密成功。分享配置：")
    for key in ("Bucket", "Region", "Prefix", "expire"):
        if key in cfg:
            print(f"  {key:8s} = {cfg[key]}")
    if "action" in cfg:
        try:
            acts = base64.b64decode(cfg["action"]).decode()
        except Exception:
            acts = cfg["action"]
        print(f"  {'action':8s} = {acts}")
    if args.show_token:
        print(f"\n原始串：{raw}")

    print(
        "\n下一步：拿到 Bucket/Prefix/Region 后仍需该分享的**签名方案**才能直连 COS 拉对象。\n"
        "若尚未突破签名，请改用浏览器打开分享链接手动下载到 data/raw/。\n"
        "下载后运行 `python scripts/02_audit.py` 会自动按列签名识别四个数据集。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
