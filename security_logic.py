"""
Access-decision logic for the multi-tenant security monitor.

A request is allowed only if it passes every check:
  1. the user and resource belong to the same tenant
  2. the user's role permits the requested action
  3. the LSTM reports the recent activity as NORMAL

Anything else, including a missing or unrecognised LSTM result, is blocked.
"""

ROLE_PERMISSIONS = {
    "Admin": ["READ", "WRITE", "DELETE"],
    "Developer": ["READ", "WRITE"],
    "Viewer": ["READ"],
}

LSTM_NORMAL = "NORMAL"
LSTM_ANOMALY = "ANOMALY"
LSTM_UNAVAILABLE = "UNAVAILABLE"


def make_security_decision(user_tenant, resource_tenant, user_role, action, lstm_status):
    """Return ("ALLOW" or "BLOCK", list of reasons for blocking)."""
    reasons = []

    action = str(action).strip().upper()
    lstm_status = str(lstm_status).strip().upper()

    # Tenant isolation
    if user_tenant != resource_tenant:
        reasons.append("Cross-Tenant Access")

    # Role-based access control (an unknown role gets no permissions)
    if action not in ROLE_PERMISSIONS.get(user_role, []):
        reasons.append("Insufficient Role Permission")

    # Anomaly detection result
    if lstm_status == LSTM_ANOMALY:
        reasons.append("LSTM Detected Suspicious Activity")
    elif lstm_status == LSTM_UNAVAILABLE:
        reasons.append("LSTM Prediction Unavailable")
    elif lstm_status != LSTM_NORMAL:
        reasons.append("Invalid LSTM Status")

    decision = "BLOCK" if reasons else "ALLOW"
    return decision, reasons
