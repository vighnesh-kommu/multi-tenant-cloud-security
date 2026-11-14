def make_security_decision(
    user_tenant,
    resource_tenant,
    user_role,
    action,
    lstm_status
):
    role_permissions = {
        "Admin": ["READ", "WRITE", "DELETE"],
        "Developer": ["READ", "WRITE"],
        "Viewer": ["READ"]
    }

    reasons = []

    if user_tenant != resource_tenant:
        reasons.append("Cross-Tenant Access")

    if action not in role_permissions.get(user_role, []):
        reasons.append("Insufficient Role Permission")

    if lstm_status == "ANOMALY":
        reasons.append("LSTM Detected Suspicious Activity")

    if lstm_status == "UNAVAILABLE":
        reasons.append("LSTM Prediction Unavailable")

    if reasons:
        return "BLOCK"

    return "ALLOW"


def test_normal_authorized_access():
    assert make_security_decision(
        "Tenant_A", "Tenant_A",
        "Admin", "READ", "NORMAL"
    ) == "ALLOW"


def test_cross_tenant_access():
    assert make_security_decision(
        "Tenant_A", "Tenant_B",
        "Admin", "READ", "NORMAL"
    ) == "BLOCK"


def test_insufficient_permission():
    assert make_security_decision(
        "Tenant_A", "Tenant_A",
        "Viewer", "WRITE", "NORMAL"
    ) == "BLOCK"


def test_lstm_anomaly():
    assert make_security_decision(
        "Tenant_A", "Tenant_A",
        "Admin", "READ", "ANOMALY"
    ) == "BLOCK"


def test_lstm_unavailable():
    assert make_security_decision(
        "Tenant_A", "Tenant_A",
        "Admin", "READ", "UNAVAILABLE"
    ) == "BLOCK"
