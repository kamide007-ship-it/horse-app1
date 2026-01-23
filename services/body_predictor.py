from __future__ import annotations
import os
import base64
import io
from pathlib import Path
from typing import Optional

from PIL import Image, ImageEnhance, ImageDraw, ImageOps
import requests

APP_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_DIR / "static"

# ==========================================================
# v1.5.7fix 画像生成（破綻しない方式）
# 1) 全体の拡大・縮小 + 中央クロップ（= タイル崩壊しない）
# 2) そこに「筋肉感」だけを固定で軽く加える（可変にしない）
#    - Contrast: 1.08
#    - Sharpness: 1.10
#
# ※芝/ダートは決めつけない（画像生成は“成長”のみ）
# ==========================================================

MUSCLE_CONTRAST = 1.08
MUSCLE_SHARPNESS = 1.10
MUSCLE_COLOR = 1.02

# ----------------------------------------------------------
# 成長倍率テーブル（ロック）
# - 入力写真の年齢に応じて「3歳像」を出す
# - 0歳→3歳（最大変化）
# - 1歳→3歳
# - 2歳→3歳
# - 3歳→4歳（微変化：任意）
# ----------------------------------------------------------
GROWTH_TABLE = {
    # 当歳→3歳は「見た目の変化が必要」なので、固定でも“差”が出る倍率にしておく
    "from_0yo_to_3yo": {"sx": 1.10, "sy": 1.28},  # 当歳→3歳：骨格＋筋肉（強め固定）
    "from_1yo_to_3yo": {"sx": 1.05, "sy": 1.12},  # 1歳→3歳：厚み中心
    "from_2yo_to_3yo": {"sx": 1.03, "sy": 1.07},  # 2歳→3歳：差は小さめ
    "from_3yo_to_4yo": {"sx": 1.02, "sy": 1.04},  # 3歳→4歳：完成に近い
}

def _stable_scale_center_crop(img: Image.Image, sx: float, sy: float) -> Image.Image:
    """拡大/縮小→中央クロップ（破綻しない）"""
    w, h = img.size
    nw = max(1, int(round(w * sx)))
    nh = max(1, int(round(h * sy)))

    resized = img.resize((nw, nh), resample=Image.BICUBIC)

    # “脚が伸びた”見え方を避けるため、縦は下基準（bottom anchor）
    # 横は中央寄せ
    canvas = Image.new("RGB", (w, h), (18, 18, 22))
    x = (w - nw) // 2
    y = h - nh
    canvas.paste(resized, (x, y))

    # もし拡大でキャンバス外にはみ出してもOK（pasteが切り落とす）
    return canvas

def _apply_muscle(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(MUSCLE_CONTRAST)
    img = ImageEnhance.Color(img).enhance(MUSCLE_COLOR)
    img = ImageEnhance.Sharpness(img).enhance(MUSCLE_SHARPNESS)
    return img


def _load_growth_reference_grid() -> Optional[bytes]:
    """成長後の“見た目”を寄せるための参照画像（最大4枚）を1枚グリッドにして返す。

    使い方:
      - 参照画像を static/ref/growth/ に置く（jpg/png/webp）
      - もしくは環境変数 GROWTH_REF_DIR で参照フォルダを指定

    OpenAI Images Edits は複数画像入力を受け付けるため、
    参照は「1枚のグリッド」にして確実に渡す。
    """

    ref_dir = os.environ.get("GROWTH_REF_DIR")
    if not ref_dir:
        # 参照画像はAI編集の“ガイド”で、UI表示用ではありません。
        # /static 配下に置くとURLで閲覧可能になるため、非公開にしたい場合は
        #   ref/growth/（staticの外）へ置けるようにしている。
        private_dir = os.path.join(APP_DIR, "ref", "growth")
        ref_dir = private_dir if os.path.isdir(private_dir) else os.path.join(STATIC_DIR, "ref", "growth")

    if not os.path.isdir(ref_dir):
        return None

    # 0) 既に ref_grid.png が用意されている場合は、それを最優先で使用
    #    （ユーザーが「理想見本4枚→ref_grid.png」を用意している運用）
    grid_file = os.path.join(ref_dir, "ref_grid.png")
    if os.path.exists(grid_file):
        try:
            with Image.open(grid_file) as im:
                im = im.convert("RGB")
                im = ImageOps.contain(im, (1024, 1024))
                buf = io.BytesIO()
                im.save(buf, format="PNG", optimize=True)
                return buf.getvalue()
        except Exception:
            # ref_grid.png が壊れている等のケースは従来ロジックへフォールバック
            pass

    exts = (".jpg", ".jpeg", ".png", ".webp")
    # ref_grid.png を自動的に拾ってしまうと「1枚しかない→見た目が弱い」になり得るので除外
    paths = [
        os.path.join(ref_dir, p)
        for p in sorted(os.listdir(ref_dir))
        if p.lower().endswith(exts) and p.lower() != "ref_grid.png"
    ]
    paths = paths[:4]
    if not paths:
        return None

    # 2x2 grid @ 1024
    cell = 512
    grid = Image.new("RGB", (1024, 1024), (18, 18, 22))
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        im = ImageOps.fit(im, (cell, cell), method=Image.BICUBIC, centering=(0.5, 0.5))
        x = (i % 2) * cell
        y = (i // 2) * cell
        grid.paste(im, (x, y))

    buf = io.BytesIO()
    grid.save(buf, format="PNG")
    return buf.getvalue()


def _build_growth_prompt(age_stage: str, coat: str) -> str:
    """タイル崩壊を避けつつ、③の見た目（自然な成長・筋肉感）に寄せるための“弱め固定”プロンプト。"""
    stage = str(age_stage).strip()
    if stage == "0":
        src = "a foal (0 year old)"
        tgt = "3 years old"
    elif stage == "2":
        src = "a 2 year old"
        tgt = "3 years old"
    elif stage == "3":
        src = "a 3 year old"
        tgt = "4 years old"
    else:
        src = "a 1 year old"
        tgt = "3 years old"

    coat_txt = coat.strip() if coat else ""
    coat_hint = f"Coat color: {coat_txt}. " if coat_txt else ""

    # 年齢段階ごとの「成長差」の強さ
    # ※ タイル崩壊を避けるため、あくまで“1枚写真”の範囲で形態変化を指示する
    if stage == "0":
        emphasis = (
            "Make the growth clearly noticeable: much taller, longer neck, longer forearm and gaskin, "
            "significantly deeper girth and stronger hindquarters, thicker topline and shoulder, overall mature racehorse body. "
        )
    elif stage == "1":
        emphasis = (
            "Make the growth noticeable: a bit taller and longer, deeper chest, more developed hindquarters and shoulder, "
            "more mature proportions. "
        )
    elif stage == "2":
        emphasis = (
            "Make the growth modest: slightly taller, a touch deeper chest, a bit more muscle definition. "
        )
    else:
        emphasis = (
            "Make the growth modest: slightly taller and longer, subtle maturity. "
        )

    return (
        "Using this exact photo as the base, produce a single realistic photograph of the SAME horse after growth. "
        f"The horse is currently {src}. Make it look approximately {tgt}. "
        "Keep the pose, camera angle, framing, background, lighting, and halter exactly the same. "
        + coat_hint +
        emphasis +
        "Increase muscle definition realistically (not exaggerated). Make the coat slightly glossier and more mature. "
        "NO collage, NO split tiles, NO extra limbs, NO distorted anatomy, NO text, NO watermark. "
        "Photorealistic, natural, seamless."
    )


def _openai_growth_edit(
    img: Image.Image,
    prompt: str,
    *,
    input_fidelity: str,
    ref_grid_png: Optional[bytes] = None,
) -> Image.Image:
    """OpenAI Images Edit を使って“弱め生成”で成長表現を付与。
    - APIキーが無い/失敗した場合は呼び出し側でフォールバックする
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = os.getenv("OPENAI_IMAGES_EDIT_URL", "https://api.openai.com/v1/images/edits")
    # 旧実装で "gpt-image-1.5" を参照していた場合、ここで失敗してフォールバックしやすい。
    # 既定は "gpt-image-1" に固定して安定運用する。
    model = os.getenv("GPT_IMAGE_MODEL", "gpt-image-1")
    size = os.getenv("GROWTH_IMAGE_SIZE", "1024x1024")
    # age_stageに応じて呼び出し側から固定値を渡す（ブレさせない）
    input_fidelity = input_fidelity
    timeout_s = int(os.getenv("GROWTH_OPENAI_TIMEOUT", "120"))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "input_fidelity": input_fidelity,
    }
    # 重要: 参照画像は「学習」ではなく、その都度“条件付け”として渡す
    #（置き場所: static/ref/growth/ に最大4枚）
    files = [
        ("image[]", ("input.png", png_bytes, "image/png")),
    ]
    if ref_grid_png:
        files.append(("image[]", ("refs.png", ref_grid_png, "image/png")))

    r = requests.post(url, headers=headers, data=data, files=files, timeout=timeout_s)
    r.raise_for_status()
    j = r.json()
    b64 = (j.get("data") or [{}])[0].get("b64_json")
    if not b64:
        raise RuntimeError("OpenAI image edit returned no b64_json")

    out = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return out

def make_3yo_prediction_image(
    side_photo_rel: Optional[str],
    coat: str = "",
    seed_key: str = "",
    age_stage: str = "1",
) -> Optional[str]:
    """実写真をベースに「成長予測イメージ」を生成（v1.5.7fix）
    - 画像生成は“破綻しない”ことを最優先
    - age_stage:
        '0' = 当歳, '1' = 1歳, '2' = 2歳, '3' = 3歳（→4歳扱い）
    """
    if not side_photo_rel:
        return None

    app_version = os.getenv("APP_VERSION", "v1.5.7fix")
    in_path = APP_DIR / side_photo_rel
    if not in_path.exists():
        return None

    out_dir = STATIC_DIR / "predictions" / app_version
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(in_path).convert("RGB")

    # 年齢ステージ→倍率テーブル
    stage = "from_1yo_to_3yo"
    if str(age_stage).strip() == "0":
        stage = "from_0yo_to_3yo"
    elif str(age_stage).strip() == "2":
        stage = "from_2yo_to_3yo"
    elif str(age_stage).strip() == "3":
        stage = "from_3yo_to_4yo"

    p = GROWTH_TABLE.get(stage, GROWTH_TABLE["from_1yo_to_3yo"])
    # 1) まずは「全体の拡大・縮小 + 中央クロップ」で必ず破綻しないベースを作る
    base = _stable_scale_center_crop(img, sx=float(p["sx"]), sy=float(p["sy"]))
    out = base

    # 参照画像（成長後の見た目）: static/ref/growth/ があれば自動で利用
    # Reference grid is optional. If it fails to load, we continue without it.
    # (Without this, the output can look like a simple zoom-only transformation.)
    status = "ZOOM"  # AIが成功すると "AI" に切り替える（UIの下部ラベルで確認できる）
    try:
        ref_grid = _load_growth_reference_grid()
    except Exception as e:
        if os.getenv("DEBUG_GROWTH", "0") == "1":
            print(f"[growth] ref grid load failed: {e}")
        ref_grid = None

    # 2) A方式：OpenAIの画像Editを“弱め固定”で一回だけかける（タイル無し）
    #    - APIキーが無い環境では自動的にスキップ（=ベース画像のまま）
    #    - 失敗しても例外で落ちずにフォールバック
    try:
        want = os.getenv("GROWTH_LOOK", "A").strip().upper()  # A/B/C... 運用用
        if want == "A" and (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
            prompt = _build_growth_prompt(str(age_stage), coat)
            # 当歳/1歳は“成長差”を出す必要があるため low（変化しやすい）固定。
            # 2歳以降は high（原型保持）固定。
            fidelity = "low" if str(age_stage).strip() in {"0", "1"} else "high"

            edited = _openai_growth_edit(out, prompt, input_fidelity=fidelity, ref_grid_png=ref_grid)

            # 生成の“効かせ具合”を固定。強くしたい当歳だけ高め。
            alpha_map = {
                "0": float(os.getenv("GROWTH_AI_ALPHA_FOAL", "0.88")),
                "1": float(os.getenv("GROWTH_AI_ALPHA_1YO", "0.70")),
                "2": float(os.getenv("GROWTH_AI_ALPHA_2YO", "0.55")),
                "3": float(os.getenv("GROWTH_AI_ALPHA_3YO", "0.45")),
            }
            a = alpha_map.get(str(age_stage).strip(), 0.55)

            # ベース（拡大クロップ）を軸にして、AI結果をブレンド → 破綻しにくい
            out = Image.blend(base, edited, max(0.0, min(1.0, a)))
            status = "AI" if ref_grid else "AI(no-ref)"
    except Exception:
        # 生成失敗は致命にしない（絶対に壊さない）
        pass

    # 3) 最後に筋肉感を“固定値”で軽く足す（ぶれない）
    out = _apply_muscle(out)

    # 下にラベル
    w, h = out.size
    canvas = Image.new("RGB", (w, h + 90), (18, 18, 22))
    canvas.paste(out, (0, 0))
    draw = ImageDraw.Draw(canvas)

    age_label = {"0": "foal(0yo)", "1": "1yo", "2": "2yo", "3": "3yo→4yo"}.get(str(age_stage).strip(), "1yo")
    text = f"Growth projection | {age_label} | {app_version} | {status} | coat={coat or '-'}"
    draw.text((18, h + 30), text, fill=(230, 230, 235))

    name = f"pred_{abs(hash(seed_key + '|' + str(age_stage))) % 10**10}.jpg"
    out_path = out_dir / name
    canvas.save(out_path, quality=92)
    return str(out_path.relative_to(APP_DIR))
