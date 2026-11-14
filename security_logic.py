
ROLE_PERMISSIONS = {
    "Admin": ["READ", "WRITE", "DELETE"],
    "Developer": ["READ", "WRITE"],
    "Viewer": ["READ"]
}


def make_security_decision(
    user_tenant,
    resource_tenant,
    user_role,
    action,
    lstm_status
):
    reasons = []

    action = str(action).strip().upper()
    lstm_status = str(lstm_status).strip().upper()

    if user_tenant != resource_tenant:
        reasons.append("Cross-Tenant Access")

    allowed_actions = ROLE_PERMISSIONS.get(user_role, [])

    if action not in allowed_actions:
        reasons.append("Insufficient Role Permission")

    if lstm_status == "ANOMALY":
        reasons.append("LSTM Detected Suspicious Activity")

    elif lstm_status == "UNAVAILABLE":
        reasons.append("LSTM Prediction Unavailable")

    elif lstm_status != "NORMAL":
        reasons.append("Invalid LSTM Status")

    decision = "BLOCK" if reasons else "ALLOW"

    return decision, reasons
