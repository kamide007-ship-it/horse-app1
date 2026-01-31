#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auction / Race AI Analyzer v2.2
- URL中心UI：URLから自動抽出できる項目はすべてURL抽出へ（オークションページ含む）
- /api/extract で URL を取得→可能な限り {馬名/性齢/血統/戦績テキスト/オークション文言} を自動入力
- /api/analyze は v2.1 の 3段指値/条件強制/競馬場順位/用途別 を維持

注意:
- サイト側の仕様変更/ブロック等で抽出できない場合は「抽出できた範囲のみ」返します。
- 過度なアクセスを避けるため、同一URLは短時間キャッシュします。
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, conlist

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
APP_TITLE = "Auction / Race AI Analyzer v2.2"
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static_v22"

TRACK_TEMPLATE = ["南関", "高知", "佐賀", "岩手", "兵庫", "笠松"]

# ------------------------------
# Simple in-memory cache
# ------------------------------
@dataclass
class CacheItem:
    ts: float
    value: Dict[str, Any]

CACHE: Dict[str, CacheItem] = {}
CACHE_TTL_SEC = 15 * 60  # 15 min

UA = os.getenv(
    "SCRAPE_UA",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
)

REQ_TIMEOUT = 15


def cache_get(key: str) -> Optional[Dict[str, Any]]:
    it = CACHE.get(key)
    if not it:
        return None
    if time.time() - it.ts > CACHE_TTL_SEC:
        CACHE.pop(key, None)
        return None
    return it.value


def cache_set(key: str, value: Dict[str, Any]) -> None:
    CACHE[key] = CacheItem(ts=time.time(), value=value)


def fetch_html(url: str) -> Tuple[str, str]:
    """
    Returns (html, content_type)
    Robust decoding: requests handles encoding; fall back to apparent_encoding.
    """
    headers = {"User-Agent": UA, "Accept-Language": "ja,en;q=0.8"}
    r = requests.get(url, headers=headers, timeout=REQ_TIMEOUT)
    ct = r.headers.get("content-type", "")
    # requests' r.text uses detected encoding; if it's broken, use apparent_encoding
    html = r.text
    if not html or ("\ufffd" in html and r.apparent_encoding):
        try:
            r.encoding = r.apparent_encoding
            html = r.text
        except Exception:
            pass
    return html, ct


def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ").replace("\u3000", " ")
    s = "\n".join([line.strip() for line in s.splitlines()])
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def is_netkeiba(url: str) -> bool:
    return "netkeiba.com" in url


def is_jbis(url: str) -> bool:
    return "jbis.or.jp" in url


def is_keibago(url: str) -> bool:
    return "keiba.go.jp" in url


def is_rakuten_auction(url: str) -> bool:
    return "auction.keiba.rakuten.co.jp" in url


def is_sat_auction(url: str) -> bool:
    return "sat-auction.jp" in url


def extract_from_keibago_debatable(url: str, html: str) -> Dict[str, Any]:
    """
    keiba.go.jp TodayRaceInfo DebaTable page
    Extract: race title/conditions/horse list snippet (as past_performance_text-like)
    """
    soup = BeautifulSoup(html, "lxml")
    title = ""
    h3 = soup.find("h3")
    if h3:
        title = clean_text(h3.get_text(" ", strip=True))
    h4 = soup.find("h4")
    if h4 and not title:
        title = clean_text(h4.get_text(" ", strip=True))

    # table-ish rows: the page is often text with sections; we take main text blocks
    main_txt = clean_text(soup.get_text("\n", strip=True))
    # make it shorter (still useful as context)
    snippet = "\n".join(main_txt.splitlines()[:220])
    return {
        "race_context": title,
        "past_performance_text": snippet,
        "source": "keiba.go.jp DebaTable",
    }


def extract_from_rakuten_auction_top(url: str, html: str) -> Dict[str, Any]:
    """
    Rakuten thoroughbred auction TOP shows list with horse cards and JBIS links.
    We'll extract the list part to a text snippet + discover JBIS links (if any).
    """
    soup = BeautifulSoup(html, "lxml")
    txt = clean_text(soup.get_text("\n", strip=True))
    snippet = "\n".join(txt.splitlines()[35:260])  # skip header clutter

    # collect JBIS links
    jbis_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "www.jbis.or.jp" in href and href not in jbis_links:
            jbis_links.append(href)
        if len(jbis_links) >= 12:
            break

    return {
        "auction_context": snippet,
        "discovered_links": {"jbis": jbis_links},
        "source": "rakuten auction top",
    }


def extract_from_netkeiba_result(url: str, html: str) -> Dict[str, Any]:
    """
    Best-effort: netkeiba result pages differ (sp / db).
    We'll grab:
      - <title> as horse_name hint
      - tables text as past_performance_text
    """
    soup = BeautifulSoup(html, "lxml")
    title = soup.find("title").get_text(strip=True) if soup.find("title") else ""
    # horse name: take first chunk before '｜' or similar
    horse_name = title.split("｜")[0].strip() if title else ""

    # Try find common header elements
    # If there is a main horse name h1
    h1 = soup.find("h1")
    if h1:
        hn = clean_text(h1.get_text(" ", strip=True))
        if hn and len(hn) <= 40:
            horse_name = hn

    # past results: tables -> text
    tables = soup.find_all("table")
    table_texts = []
    for t in tables[:3]:
        table_texts.append(clean_text(t.get_text("\n", strip=True)))
    past = "\n\n".join([t for t in table_texts if t])

    # fallback: body snippet
    if not past:
        body = clean_text(soup.get_text("\n", strip=True))
        past = "\n".join(body.splitlines()[:220])

    return {
        "horse_name": horse_name,
        "past_performance_text": past,
        "source": "netkeiba result (best-effort)",
    }


def extract_from_jbis_record_or_basic(url: str, html: str) -> Dict[str, Any]:
    """
    Best-effort for JBIS pages: try to pull horse name, sex/age maybe in header,
    and a main text snippet as past_performance_text.
    """
    soup = BeautifulSoup(html, "lxml")
    title = soup.find("title").get_text(strip=True) if soup.find("title") else ""
    horse_name = title.split("｜")[0].strip() if title else ""
    h1 = soup.find("h1")
    if h1:
        hn = clean_text(h1.get_text(" ", strip=True))
        if hn and len(hn) <= 40:
            horse_name = hn

    text = clean_text(soup.get_text("\n", strip=True))
    snippet = "\n".join(text.splitlines()[:240])

    return {
        "horse_name": horse_name,
        "past_performance_text": snippet,
        "source": "jbis (best-effort)",
    }


def extract_from_sat_auction(url: str, html: str) -> Dict[str, Any]:
    """
    SAT Auction is often JS-driven; best-effort: return visible text snippet.
    """
    soup = BeautifulSoup(html, "lxml")
    txt = clean_text(soup.get_text("\n", strip=True))
    snippet = "\n".join(txt.splitlines()[:220])
    return {"auction_context": snippet, "source": "sat-auction (best-effort)"}


def merge_extract(base: Dict[str, Any], add: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in add.items():
        if v is None or v == "":
            continue
        if k in ("discovered_links",):
            out.setdefault(k, {})
            for kk, vv in (v or {}).items():
                if kk not in out[k]:
                    out[k][kk] = vv
                else:
                    # merge unique
                    cur = out[k][kk]
                    if isinstance(cur, list) and isinstance(vv, list):
                        for x in vv:
                            if x not in cur:
                                cur.append(x)
        else:
            # if already exists, append for text fields
            if k in ("auction_context", "past_performance_text") and out.get(k):
                out[k] = (out[k] + "\n\n" + str(v)).strip()
            else:
                out[k] = v
    return out


# ------------------------------
# AI output schema (same as v2.1)
# ------------------------------
class TrackRank(BaseModel):
    track: str
    fit_score: float = Field(..., ge=0.0, le=1.0)
    reason: str


class AnalyzeResponse(BaseModel):
    bid_ideal_man: Optional[float] = None
    bid_max_man: Optional[float] = None
    bid_walkaway_man: Optional[float] = None

    buy_decision: Literal["BUY", "HOLD", "PASS"]
    decision_confidence: float = Field(..., ge=0.0, le=1.0)

    purpose_axis: Literal["勝ち上がり", "稼ぐ", "繁殖"] = "稼ぐ"

    best_distance: str = ""
    running_style: str = ""

    track_ranking: List[TrackRank] = Field(default_factory=list)

    earnings_outlook: str = ""
    breeding_outlook: str = ""

    key_risks: List[str] = Field(default_factory=list)

    buy_conditions: conlist(str, min_length=2, max_length=2) = Field(default_factory=list)
    pass_conditions: conlist(str, min_length=2, max_length=2) = Field(default_factory=list)

    reasoning_bullets: List[str] = Field(default_factory=list)
    what_to_confirm_next: List[str] = Field(default_factory=list)


SYSTEM = f"""あなたは日本の競馬（JRA/NAR）とサラブレッド取引に詳しい、実務の購買判断アナリスト。
ユーザーは「自分だけが使う」前提。結論は短く、しかし“条件付き”で現実的に提示する。
外部サイトへアクセス/スクレイピングはするが、ユーザーが入力したURLに対してのみ行う。
不確実な点は不確実と明示し、confidenceを下げる。

【出力制約（絶対）】
- 出力は必ず指定JSONスキーマに一致（Structured Outputs）。JSON以外は一切出さない。
- bid_ideal_man / bid_max_man / bid_walkaway_man を可能な範囲で埋める（万円）。根拠が薄い場合はnullでも可だが、原則埋める。
- buy_decision が BUY/HOLD/PASS のいずれでも、buy_conditions と pass_conditions は必ず「2つずつ」埋める（空配列禁止）。
- track_ranking はテンプレ競馬場を優先して順位付け：{", ".join(TRACK_TEMPLATE)}（必要なら最後に「その他」も可）。
  fit_score は0〜1の相対。reasonは短く（1文）。
- purpose_axis（勝ち上がり/稼ぐ/繁殖）に合わせて評価軸を切替。
  * 勝ち上がり: 上のクラスに上がれる再現性（脚質×距離×馬場×負け方）
  * 稼ぐ: 回収しやすさ/相手関係/賞金設計/取りこぼしにくさ
  * 繁殖: 母系の価値/産駒の売りやすさ/適性の遺伝しやすさ/故障リスク

【固定ロジック】
1) 目的は3点：①指値（3段）②買い/見送り（条件提示）③適性競馬場・距離・脚質
2) 価格の歪み重視：中央未勝利・少戦数で安い場合、地方で回収余地（ただし負け方が致命的なら除外）
3) 場替わり再現性重視：ある場で勝っても別場で再現しないタイプは注意（HOLD/PASS寄り、条件提示）
4) 画像/動画フレームがあれば、歩様/肢勢/左右差/推進の癖を所見→リスク/適性に落とす（断定しすぎない）
"""


def b64_data_url_from_bytes(content: bytes, mime: str) -> str:
    b64 = base64.b64encode(content).decode("ascii")
    return f"data:{mime};base64,{b64}"


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_frames_ffmpeg(video_path: str, max_frames: int = 4) -> List[bytes]:
    if not ffmpeg_exists():
        return []
    frames: List[bytes] = []
    with tempfile.TemporaryDirectory() as td:
        out_pattern = os.path.join(td, "frame_%02d.jpg")
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", "fps=1", "-q:v", "3", out_pattern]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        for i in range(1, 99):
            p = os.path.join(td, f"frame_{i:02d}.jpg")
            if not os.path.exists(p):
                break
            frames.append(Path(p).read_bytes())
            if len(frames) >= max_frames:
                break
    return frames


app = FastAPI(title=APP_TITLE)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/extract")
def api_extract(urls: str = Form(...)):
    """
    urls: newline separated
    Returns extracted fields to auto-fill UI.
    """
    urls_list = [u.strip() for u in urls.splitlines() if u.strip()]
    if not urls_list:
        raise HTTPException(status_code=400, detail="urls is empty")

    out: Dict[str, Any] = {"horse_url": "\n".join(urls_list)}
    sources = []

    for u in urls_list[:6]:  # safety cap
        cached = cache_get(u)
        if cached is None:
            try:
                html, ct = fetch_html(u)
                add: Dict[str, Any] = {}
                if is_keibago(u) and "DebaTable" in u:
                    add = extract_from_keibago_debatable(u, html)
                elif is_rakuten_auction(u):
                    add = extract_from_rakuten_auction_top(u, html)
                elif is_sat_auction(u):
                    add = extract_from_sat_auction(u, html)
                elif is_jbis(u):
                    add = extract_from_jbis_record_or_basic(u, html)
                elif is_netkeiba(u):
                    add = extract_from_netkeiba_result(u, html)
                else:
                    # generic
                    soup = BeautifulSoup(html, "lxml")
                    txt = clean_text(soup.get_text("\n", strip=True))
                    add = {"auction_context": "\n".join(txt.splitlines()[:220]), "source": "generic"}
                cached = add
                cache_set(u, cached)
            except Exception as e:
                cached = {"source": "error", "error": str(e)}
        sources.append({ "url": u, **{k:v for k,v in cached.items() if k in ("source","error")} })
        out = merge_extract(out, cached)

    out["sources"] = sources
    return JSONResponse(content={"ok": True, "data": out})


@app.post("/api/analyze")
async def analyze(
    mode: str = Form("auction"),
    horse_url: str = Form(""),
    horse_name: str = Form(""),
    age: Optional[str] = Form(None),
    sex: str = Form(""),
    sire: str = Form(""),
    dam: str = Form(""),
    dam_sire: str = Form(""),
    purpose_axis: str = Form("稼ぐ"),
    constraints: str = Form(""),
    auction_context: str = Form(""),
    past_performance_text: str = Form(""),
    gait_notes: str = Form(""),
    target_tracks: str = Form(""),
    max_budget_man: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set on server.")
    if OpenAI is None:
        raise HTTPException(status_code=500, detail="openai SDK not installed. pip install openai")

    if purpose_axis not in ("勝ち上がり", "稼ぐ", "繁殖"):
        purpose_axis = "稼ぐ"

    def _safe_int(x: Optional[str]) -> Optional[int]:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        try:
            return int(float(s))
        except Exception:
            return None

    def _safe_float(x: Optional[str]) -> Optional[float]:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        try:
            return float(s)
        except Exception:
            return None

    age_i = _safe_int(age)
    budget_f = _safe_float(max_budget_man)

    user_text = f"""# 入力（URL自動抽出＋ユーザー追記）
モード: {mode}
評価軸: {purpose_axis}

馬URL（参照元）:
{horse_url}

馬名: {horse_name}
年齢: {age_i}
性別: {sex}
父: {sire}
母: {dam}
母父: {dam_sire}

制約: {constraints}
予算上限（万円）: {budget_f}

オークション情報/募集文（URL抽出＋追記）:
{auction_context or '(なし)'}

過去走（URL抽出＋コピペ）:
{past_performance_text or '(なし)'}

歩様メモ（任意）:
{gait_notes or '(なし)'}

想定競馬場/地域（任意）:
{target_tracks or '(なし)'}

（注意）競馬場ランキングはテンプレ（{", ".join(TRACK_TEMPLATE)}）を優先して順位付けしてください。
"""

    mm_input: List[Dict[str, Any]] = [{"type": "input_text", "text": user_text}]

    if photo is not None:
        content = await photo.read()
        mime = photo.content_type or "image/jpeg"
        if not mime.startswith("image/"):
            raise HTTPException(status_code=400, detail="photo must be an image.")
        mm_input.append({"type": "input_text", "text": f"添付写真: {photo.filename}"})
        mm_input.append({"type": "input_image", "image_url": b64_data_url_from_bytes(content, mime)})

    if video is not None:
        if not (video.content_type or "").startswith("video/"):
            raise HTTPException(status_code=400, detail="video must be a video file.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename or "clip.mp4").suffix) as tf:
            tf.write(await video.read())
            tmp_video_path = tf.name
        try:
            frames = extract_frames_ffmpeg(tmp_video_path, max_frames=4)
            if frames:
                mm_input.append({"type": "input_text", "text": f"添付動画: {video.filename}（フレーム抽出 {len(frames)}枚）"})
                for b in frames:
                    mm_input.append({"type": "input_image", "image_url": b64_data_url_from_bytes(b, "image/jpeg")})
            else:
                mm_input.append({"type": "input_text", "text": f"添付動画: {video.filename}（注：ffmpegなし→フレーム抽出不可。歩様メモ重視で推定）"})
        finally:
            try:
                os.remove(tmp_video_path)
            except Exception:
                pass

    client = OpenAI(api_key=api_key)
    schema: Dict[str, Any] = AnalyzeResponse.model_json_schema()

    def enforce_additional_properties_false(node: Any) -> None:
        if isinstance(node, dict):
            # Strict JSON schema requires additionalProperties=false for objects
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
            for v in node.values():
                enforce_additional_properties_false(v)
        elif isinstance(node, list):
            for item in node:
                enforce_additional_properties_false(item)

    enforce_additional_properties_false(schema)


    try:
        resp = client.responses.create(
            model=DEFAULT_MODEL,
            input=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": mm_input},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "auction_race_analysis_v22",
                    "schema": schema,
                    "strict": True,
                }
            },
        )

        out_text = None
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    out_text = getattr(c, "text", None)
                    if out_text:
                        break
            if out_text:
                break
        if not out_text:
            raise RuntimeError("No output_text in response")

        return JSONResponse(content={"ok": True, "data": out_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analyze failed: {e}")


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": DEFAULT_MODEL}
