from __future__ import annotations
import os, secrets
from typing import Optional
from extensions import db
from models import PaymentRequest, User

def bank_info() -> dict:
    return {
        "name": os.getenv("BANK_NAME","楽天銀行"),
        "branch": os.getenv("BANK_BRANCH","エンカ支店"),
        "type": os.getenv("BANK_ACCOUNT_TYPE","普通"),
        "number": os.getenv("BANK_ACCOUNT_NUMBER","1546960"),
        "account_name": os.getenv("BANK_ACCOUNT_NAME","カミデケンタロウ"),
        "note": os.getenv("BANK_NOTE","振込名義の末尾に参照コードを入れてください"),
    }

def create_bank_payment_request(user: User, plan: str) -> Optional[PaymentRequest]:
    ref = secrets.token_hex(4)
    pr = PaymentRequest(user_id=user.id, plan=plan, reference_code=ref, status="pending")
    db.session.add(pr)
    db.session.commit()
    return pr

def approve_payment_request(pr: PaymentRequest) -> None:
    u = User.query.get(pr.user_id)
    if not u: return
    u.plan = pr.plan
    u.quota_used_month = 0
    u.quota_month = ""
    pr.status = "approved"
    db.session.commit()
