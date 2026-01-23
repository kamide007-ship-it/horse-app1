from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from extensions import db, login_manager
from models import Evaluation, PaymentRequest, User
from services.evaluator import evaluate_horse
from services.body_predictor import make_3yo_prediction_image
from services.market import estimate_market
from services.bank_payments import approve_payment_request, bank_info, create_bank_payment_request

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _parse_number(value: str | None) -> float | None:
    """Parse numeric user inputs robustly (commas/units allowed).

    Examples accepted:
      - "920" / "920.5"
      - "920,000"
      - "920万円" / "920 万円" / "920万"
      - "920000円"
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # normalize
    s = s.replace(",", "")
    s = s.replace("円", "")
    s = s.replace("万円", "")
    s = s.replace("万", "")
    s = s.replace(" ", "")
    # keep only the first numeric token
    import re

    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or os.environ.get("SECRET_KEY") or "dev-secret"
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///app.db")
    if app.config["SQLALCHEMY_DATABASE_URI"].startswith("postgres://"):
        app.config["SQLALCHEMY_DATABASE_URI"] = app.config["SQLALCHEMY_DATABASE_URI"].replace("postgres://", "postgresql://", 1)
    if app.config["SQLALCHEMY_DATABASE_URI"].startswith("postgresql://"):
        app.config["SQLALCHEMY_DATABASE_URI"] = app.config["SQLALCHEMY_DATABASE_URI"].replace("postgresql://", "postgresql+psycopg://", 1)
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    login_manager.init_app(app)

    @app.context_processor
    def inject_app_version():
        return {"APP_VERSION": os.getenv("APP_VERSION", "v1.5.7fix")}

    @app.template_filter("yen")
    def _yen(v) -> str:
        try:
            n = float(v or 0)
        except Exception:
            n = 0.0
        return f"{int(round(n)):,}"

    with app.app_context():
        db.create_all()
        admin_email = (os.environ.get("ADMIN_EMAIL") or "").strip().lower()
        admin_password = os.environ.get("ADMIN_PASSWORD") or ""
        if admin_email and admin_password:
            a = User.query.filter_by(email=admin_email).first()
            if a is None:
                a = User(email=admin_email, plan="enterprise", is_admin=True)
                a.set_password(admin_password)
                db.session.add(a)
                db.session.commit()

    @login_manager.user_loader
    def load_user(user_id: str):
        return db.session.get(User, int(user_id))

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "ts": datetime.utcnow().isoformat()}

    # ---- Auth ----
    @app.route("/register", methods=["GET", "POST"])
    def auth_register():
        if request.method == "POST":
            email = (request.form.get("email") or "").strip().lower()
            password = request.form.get("password") or ""
            if not email or not password:
                flash("メールとパスワードを入力してください。")
                return render_template("register.html")
            if User.query.filter_by(email=email).first():
                flash("そのメールは既に登録されています。")
                return render_template("register.html")
            u = User(email=email)
            u.set_password(password)
            db.session.add(u)
            db.session.commit()
            login_user(u)
            return redirect(url_for("index"))
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def auth_login():
        if request.method == "POST":
            email = (request.form.get("email") or "").strip().lower()
            password = request.form.get("password") or ""
            u = User.query.filter_by(email=email).first()
            if u and u.check_password(password):
                login_user(u)
                return redirect(url_for("index"))
            flash("ログインに失敗しました。")
        return render_template("login.html")

    @app.get("/logout")
    def auth_logout():
        logout_user()
        return redirect(url_for("auth_login"))

    # ---- Pricing ----
    @app.get("/pricing")
    @login_required
    def pricing():
        return render_template("pricing.html")

    # ---- Admin ----
    @app.get("/admin")
    @login_required
    def admin_console():
        if not getattr(current_user, "is_admin", False):
            abort(403)
        users = User.query.order_by(User.id.asc()).all()
        requests = PaymentRequest.query.order_by(PaymentRequest.created_at.desc()).limit(200).all()
        return render_template("admin.html", users=users, requests=requests)

    @app.post("/checkout/<plan>")
    @login_required
    def checkout(plan: str):
        if plan == "free":
            current_user.plan = "free"
            db.session.commit()
            return redirect(url_for("index"))
        pr = create_bank_payment_request(user=current_user, plan=plan)
        if pr is None:
            flash("プラン申請の作成に失敗しました。時間をおいて再度お試しください。")
            return redirect(url_for("pricing"))
        return render_template("bank_transfer.html", plan=plan, request_ref=pr.reference_code, bank=bank_info())

    # ---- Main ----
    @app.route("/", methods=["GET", "POST"])
    @login_required
    def index():
        if request.method == "GET":
            return render_template("input.html")

        if not current_user.can_eval():
            return render_template("upgrade.html", plan=current_user.plan)

        form = request.form
        coat = (form.get("coat") or "").strip()
        coat_other = (form.get("coat_other") or "").strip()
        if "その他" in coat and coat_other:
            coat = coat_other

        def _get_any(*keys: str) -> str:
            """HTML側の name 変更に強くする（過去バージョン互換）。"""
            for k in keys:
                v = form.get(k)
                if v is not None and str(v).strip() != "":
                    return v
            return ""

        payload = {
            "sire": form.get("sire", ""),
            "dam": form.get("dam", ""),
            "damsire": form.get("damsire", ""),
            "dob": form.get("dob", ""),
            "sex": form.get("sex", ""),
            "coat": coat,
            "age_stage": _get_any("age_stage", "age", "stage") or "1",
            "body_weight": _get_any("body_weight", "weight", "bw"),
            # 測尺：HTML側の name が (height_cm/chest_cm/cannon_cm) でも (height/girth/cannon) でもOK
            "height": _get_any("height_cm", "height", "withers", "wh"),
            "girth": _get_any("chest_cm", "girth", "chest", "heartgirth"),
            "cannon": _get_any("cannon_cm", "cannon", "cannon_bone", "cannoncirc"),
            "distance_m": form.get("distance_m", ""),
            "notes": form.get("notes", ""),
        }

        
        def _num_str(x: str) -> str:
            s = str(x or "").strip().replace(",", "")
            # pick first number like 155.5
            import re as _re
            m = _re.search(r"(\d+(?:\.\d+)?)", s)
            return m.group(1) if m else ""

        def _norm_cm(val: str, mm_threshold: float, decimals: int = 1) -> str:
            s = _num_str(val)
            if not s:
                return ""
            try:
                n = float(s)
            except Exception:
                return ""
            # mm入力対策（例: 1600 -> 160.0）
            if n >= mm_threshold:
                n = n / 10.0
            # 0や極端値は空扱い（評価の暴走防止）
            if n <= 0:
                return ""
            fmt = f"{{:.{decimals}f}}"
            return fmt.format(n).rstrip("0").rstrip(".")

        # --- 測尺入力の自動補正（cm想定 / mm混入を吸収） ---
        payload["height"] = _norm_cm(payload.get("height",""), mm_threshold=350, decimals=1)   # 1600など
        payload["girth"]  = _norm_cm(payload.get("girth",""),  mm_threshold=350, decimals=1)
        payload["cannon"] = _norm_cm(payload.get("cannon",""), mm_threshold=60,  decimals=1)   # 200など
        market_inputs = {
            "sire_fee_median": form.get("sire_fee_median", ""),
            "dam_value": form.get("dam_value", ""),
            "blacktype_count": form.get("blacktype_count", ""),
            "nearby_gsw": form.get("nearby_gsw", ""),
            "market_price_avg_man": form.get("market_price_avg_man",""),
        }

        def save_upload(file_storage, suffix: str) -> str | None:
            if not file_storage or not getattr(file_storage, "filename", ""):
                return None
            name = f"{int(datetime.utcnow().timestamp())}_{current_user.id}_{suffix}"
            ext = Path(file_storage.filename).suffix.lower()[:10] or ".bin"
            out = UPLOAD_DIR / f"{name}{ext}"
            file_storage.save(out)
            return str(out.relative_to(APP_DIR))

        side_path = save_upload(request.files.get("side_photo"), "side")
        video_path = save_upload(request.files.get("video"), "video")

        payload["_side_rel"] = side_path or ""

        scores = evaluate_horse(payload=payload, side_photo_rel=side_path, video_rel=video_path)
        market = estimate_market(payload=payload, market_inputs=market_inputs)

        seed_key = f"{payload.get('sire','')}-{payload.get('dam','')}-{payload.get('damsire','')}-{payload.get('dob','')}-{payload.get('sex','')}-{payload.get('coat','')}"
        try:
            pred_rel = make_3yo_prediction_image(side_photo_rel=side_path, coat=payload.get("coat",""), seed_key=seed_key, age_stage=str(payload.get("age_stage","1")))
        except Exception:
            pred_rel = None

        result = {
            "scores": scores,
            "market": market,
            "inputs_used": {
                "has_side_photo": bool(side_path),
                "has_video": bool(video_path),
                "note_ja": "空欄でも評価できますが、測尺/距離/側面写真/動画が揃うほど確信度が上がります。",
            },
            "predicted_3yo_rel": pred_rel,
        }

        label = f"{payload.get('sire','')} × {payload.get('dam','')}"
        ev = Evaluation(
            user_id=current_user.id,
            horse_label=label,
            input_json=json.dumps({"payload": payload, "market": market_inputs}, ensure_ascii=False),
            result_json=json.dumps(result, ensure_ascii=False),
            side_photo_path=side_path,
            video_path=video_path,
            predicted_3yo_path=pred_rel,
        )
        db.session.add(ev)
        current_user.consume_eval()
        db.session.commit()

        return render_template("result.html", payload=payload, result=result, label=label)

    @app.get("/history")
    @login_required
    def history():
        current_user.refresh_monthly_counter()
        if current_user.plan == "free":
            used = int(current_user.quota_used_total or 0)
            limit = 1
        else:
            used = int(current_user.quota_used_month or 0)
            limit = current_user.monthly_limit()

        rows = Evaluation.query.filter_by(user_id=current_user.id).order_by(Evaluation.created_at.desc()).limit(50).all()
        items = []
        for r in rows:
            try:
                res = json.loads(r.result_json)
                total = int((res.get("scores") or {}).get("total", 0))
                rank = (res.get("scores") or {}).get("rank", "-")
            except Exception:
                total, rank = 0, "-"
            items.append({"id": r.id, "created_at": r.created_at, "horse_label": r.horse_label, "total": total, "rank": rank})
        return render_template("history.html", items=items, used=used, limit=limit)

    @app.get("/e/<int:eval_id>")
    @login_required
    def view_evaluation(eval_id: int):
        r = Evaluation.query.filter_by(id=eval_id, user_id=current_user.id).first_or_404()
        data = json.loads(r.input_json)
        payload = data.get("payload") or {}
        result = json.loads(r.result_json)
        label = r.horse_label
        return render_template("result.html", payload=payload, result=result, label=label)

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
