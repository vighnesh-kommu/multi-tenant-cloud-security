"""Central security checks for the multi-tenant cloud application."""

ROLE_PERMISSIONS = {
    "Admin": {"READ", "WRITE", "DELETE"},
    "Developer": {"READ", "WRITE"},
    "Viewer": {"READ"},
}

VALID_TENANTS = {"Tenant_A", "Tenant_B"}
VALID_ACTIONS = {"READ", "WRITE", "DELETE"}
VALID_LSTM_STATUSES = {"NORMAL", "ANOMALY", "UNAVAILABLE"}


def make_security_decision(
    user_tenant,
    resource_tenant,
    user_role,
    action,
    lstm_status,
):
    """Return ALLOW or BLOCK with the reasons for the decision."""

    reasons = []

    user_tenant = str(user_tenant or "").strip()
    resource_tenant = str(resource_tenant or "").strip()
    user_role = str(user_role or "").strip()
    action = str(action or "").strip().upper()
    lstm_status = str(lstm_status or "").strip().upper()

    # 1. Validate tenant information.
    if user_tenant not in VALID_TENANTS:
        reasons.append("Invalid User Tenant")

    if resource_tenant not in VALID_TENANTS:
        reasons.append("Invalid Resource Tenant")

    # 2. Prevent access across tenant boundaries.
    if (
        user_tenant in VALID_TENANTS
        and resource_tenant in VALID_TENANTS
        and user_tenant != resource_tenant
    ):
        reasons.append("Cross-Tenant Access")

    # 3. Check role-based permissions.
    allowed_actions = ROLE_PERMISSIONS.get(user_role)

    if allowed_actions is None:
        reasons.append("Invalid User Role")
    elif action not in allowed_actions:
        reasons.append("Insufficient Role Permission")

    if action not in VALID_ACTIONS:
        reasons.append("Invalid Requested Action")

    # 4. Check the LSTM anomaly detection result.
    if lstm_status == "ANOMALY":
        reasons.append("LSTM Detected Suspicious Activity")

    elif lstm_status == "UNAVAILABLE":
        reasons.append("LSTM Prediction Unavailable")

    elif lstm_status not in VALID_LSTM_STATUSES:
        reasons.append("Invalid LSTM Status")

    # Fail closed: any security issue results in BLOCK.
    decision = "BLOCK" if reasons else "ALLOW"

    return decision, reasons
