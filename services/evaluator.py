from __future__ import annotations
import math
from typing import Dict, Any, Optional

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None: return default
        s = str(v).strip()
        if not s: return default
        return float(s)
    except Exception:
        return default

def _class_words(score_0_100: float) -> str:
    # "厳しめ"の言葉に寄せる
    if score_0_100 >= 82: return "JRAオープン級（相当上振れ）"
    if score_0_100 >= 75: return "地方重賞上位〜JRA1勝級の壁"
    if score_0_100 >= 68: return "地方A級〜B上位（条件が揃えば重賞圏）"
    if score_0_100 >= 60: return "地方B級〜C1上位（展開/馬場で上振れ）"
    if score_0_100 >= 52: return "地方C級〜B下位（まずは条件戦）"
    return "未勝利〜地方C級（成長待ち）"

def evaluate_horse(payload: Dict[str, Any], side_photo_rel: Optional[str]=None, video_rel: Optional[str]=None) -> Dict[str, Any]:
    """v1.5.6-hotfix: 安定稼働を優先した簡易評価（落ちない・数値が現実的）。
    - ここは v2.0 で gopaddock-video-ai へ差し替え前提
    """
    bw = _to_float(payload.get("body_weight"), 450.0)
    h  = _to_float(payload.get("height"), 155.0)
    g  = _to_float(payload.get("girth"), 175.0)
    c  = _to_float(payload.get("cannon"), 19.5)
    dist = _to_float(payload.get("distance_m"), 1600.0)

    # ベース(厳しめ) 50±
    base = 50.0
    base += _clamp((bw-450.0)/6.0, -8, 8)     # 体重
    base += _clamp((h-155.0)*0.6, -6, 6)      # 体高
    base += _clamp((g-175.0)*0.25, -6, 6)     # 胸囲
    base += _clamp((c-19.5)*1.0, -6, 6)       # 管囲

    # 距離適性(中距離付近をニュートラル)
    dist_penalty = abs(dist-1600.0)/400.0     # 400m差で+1
    stamina_bias = _clamp(6.0 - dist_penalty*2.0, -6, 6)

    has_video = bool(video_rel)
    has_photo = bool(side_photo_rel)

    # 「動画があるほど精度UP。ただし点数は上げ過ぎない」
    video_bonus = 2.0 if has_video else 0.0
    photo_bonus = 1.0 if has_photo else 0.0

    total = _clamp(base + video_bonus + photo_bonus, 35, 88)

    # スピード/スタミナは total を分解（上限を絞る）
    # turfiness は断定せず傾向表示：体型から「スピード寄り度」をゆるく推定
    speediness = _sigmoid(0.08*((bw-450.0) + (h-155.0)*4.0 - (g-175.0)*2.0))
    turfiness = speediness  # 今は同義（v2.0で差し替え）

    speed = _clamp(total + (speediness-0.5)*10.0, 30, 90)
    stamina = _clamp(total + stamina_bias + (0.5-speediness)*8.0, 30, 90)
    power = _clamp(total + (bw-450.0)/10.0, 30, 90)
    durability = _clamp(55.0 + (c-19.5)*2.5 + (g-175.0)*0.1, 30, 90)
    risk = _clamp(60.0 - (durability-55.0) + (0 if has_video else 4.0), 10, 90)

    # 確信度（%）
    conf = 55.0
    conf += 18.0 if has_video else 0.0
    conf += 10.0 if has_photo else 0.0
    # 測尺が埋まっているほど増加
    filled = sum(1 for k in ("body_weight","height","girth","cannon","distance_m") if str(payload.get(k,"")).strip())
    conf += filled*3.0
    conf = _clamp(conf, 35, 92)

    return {
        "total": int(round(total)),
        "rank": _class_words(total),
        "confidence_pct": int(round(conf)),
        "speed": int(round(speed)),
        "stamina": int(round(stamina)),
        "power": int(round(power)),
        "durability": int(round(durability)),
        "risk": int(round(risk)),
        "turfiness": round(float(turfiness), 3),
        "quality": {
            "has_video": 1.0 if has_video else 0.0,
            "has_photo": 1.0 if has_photo else 0.0,
            "filled_measurements": float(filled),
        },
        "growth_vector": {
            "estimated_growth": "画像生成が無い場合は、測尺と年齢で推定(簡易)。",
            "notes": "v2.0で学習モデルへ置換予定",
        },
    }
