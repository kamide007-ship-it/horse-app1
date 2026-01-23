from __future__ import annotations
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db

class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    plan = db.Column(db.String(32), default="free", nullable=False)  # free/starter/pro/enterprise
    quota_used_total = db.Column(db.Integer, default=0, nullable=False)
    quota_used_month = db.Column(db.Integer, default=0, nullable=False)
    quota_month = db.Column(db.String(7), default="", nullable=False)  # YYYY-MM
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def set_password(self, pw: str) -> None:
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw: str) -> bool:
        return check_password_hash(self.password_hash, pw)

    def monthly_limit(self) -> int | None:
        if self.plan == "free": return 1
        if self.plan == "starter": return 10
        if self.plan == "pro": return 30
        if self.plan == "enterprise": return None
        return 1

    def refresh_monthly_counter(self, now: datetime | None = None) -> None:
        now = now or datetime.utcnow()
        key = f"{now.year:04d}-{now.month:02d}"
        if self.quota_month != key:
            self.quota_month = key
            self.quota_used_month = 0

    def can_eval(self) -> bool:
        self.refresh_monthly_counter()
        if self.plan == "free":
            return self.quota_used_total < 1
        limit = self.monthly_limit()
        return True if limit is None else (self.quota_used_month < limit)

    def consume_eval(self) -> None:
        self.refresh_monthly_counter()
        self.quota_used_total += 1
        self.quota_used_month += 1

class Evaluation(db.Model):
    __tablename__ = "evaluations"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    horse_label = db.Column(db.String(255), default="", nullable=False)
    input_json = db.Column(db.Text, nullable=False)
    result_json = db.Column(db.Text, nullable=False)
    side_photo_path = db.Column(db.String(512), nullable=True)
    video_path = db.Column(db.String(512), nullable=True)
    predicted_3yo_path = db.Column(db.String(512), nullable=True)
    user = db.relationship("User", backref=db.backref("evaluations", lazy=True))

class PaymentRequest(db.Model):
    __tablename__ = "payment_requests"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    plan = db.Column(db.String(32), nullable=False)
    reference_code = db.Column(db.String(32), unique=True, nullable=False)
    status = db.Column(db.String(16), default="pending", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    decided_at = db.Column(db.DateTime, nullable=True)
    user = db.relationship("User", backref=db.backref("payment_requests", lazy=True))
