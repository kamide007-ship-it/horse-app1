from __future__ import annotations
from typing import Dict, Any

def _to_float(v: Any, default: float=0.0) -> float:
    try:
        if v is None: return default
        s = str(v).strip()
        if not s: return default
        # allow commas and simple units ("円", "万円", "万")
        s = s.replace(",", "").replace(" ", "")
        s = s.replace("万円", "").replace("万", "").replace("円", "")
        # keep only first numeric token
        import re
        m = re.search(r"[-+]?(?:\d+\.?\d*|\d*\.?\d+)", s)
        if not m:
            return default
        return float(m.group(0))
    except Exception:
        return default

def estimate_market(payload: Dict[str, Any], market_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """価格は簡易推定。ブラックタイプ数/近親GSW数は
    v1.5.6 では「URL貼り付け→後でAI抽出」へ移行準備（数値も可）。
    """
    sire_fee = _to_float(market_inputs.get("sire_fee_median"), 0.0)
    dam_val = _to_float(market_inputs.get("dam_value"), 0.0)

    bt = market_inputs.get("blacktype_count","")
    gsw = market_inputs.get("nearby_gsw","")
    avg_man_raw = market_inputs.get("market_price_avg_man","")

    bt_n = _to_float(bt, 0.0) if str(bt).strip() and str(bt).strip().replace('.','',1).isdigit() else None
    gsw_n = _to_float(gsw, 0.0) if str(gsw).strip() and str(gsw).strip().replace('.','',1).isdigit() else None

    base = max(800000, sire_fee*3 + dam_val*0.6)
    bonus = 0.0
    if bt_n is not None: bonus += bt_n*200000
    if gsw_n is not None: bonus += gsw_n*350000

    low = base*0.8 + bonus*0.7
    high = base*1.3 + bonus*1.2

    # 市場価格平均値（万円）入力がある場合は、推定レンジに必ず反映する
    # 入力ゆれ対策:
    #  - 920 (万円) のような「万円」前提
    #  - 920000 (円) / "¥920,000" / "920000円" などが来ても自動で万円へ補正
    avg_val = _to_float(avg_man_raw, 0.0)
    raw_s = str(avg_man_raw or "").strip()
    has_man = ("万" in raw_s) or ("万円" in raw_s)
    has_yen = ("円" in raw_s) or ("¥" in raw_s)
    if avg_val > 0 and (has_yen or (avg_val >= 10000 and not has_man)):
        avg_man = avg_val / 10000.0
    else:
        avg_man = avg_val
    avg_yen = int(round(avg_man * 10000)) if avg_man > 0 else 0
    if avg_yen > 0:
        # 推定値と平均値の“ハイブリッド”（入力を無視しない）
        low = low*0.6 + (avg_yen*0.80)*0.4
        high = high*0.6 + (avg_yen*1.20)*0.4

    return {
        "yen_low": int(round(low)),
        "yen_high": int(round(high)),
        "market_avg": {"man": float(avg_man) if avg_yen>0 else None, "yen": int(avg_yen) if avg_yen>0 else None},
        "inputs": {
            "blacktype": {"raw": bt, "mode": "url_or_number"},
            "nearby_gsw": {"raw": gsw, "mode": "url_or_number"},
        },
        "note": "v2.0でURLからの自動抽出(AI)に対応予定。現状は数値入力が最も安定。市場価格平均値（万円）を入れると推定に反映します。",
    }
