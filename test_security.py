
from security_logic import make_security_decision


def test_normal_authorized_access():
    decision, reasons = make_security_decision(
        "Tenant_A", "Tenant_A",
        "Admin", "READ", "NORMAL"
    )
    assert decision == "ALLOW"
    assert reasons == []


def test_cross_tenant_access():
    decision, reasons = make_security_decision(
        "Tenant_A", "Tenant_B",
        "Admin", "READ", "NORMAL"
    )
    assert decision == "BLOCK"
    assert "Cross-Tenant Access" in reasons


def test_insufficient_permission():
    decision, reasons = make_security_decision(
        "Tenant_A", "Tenant_A",
        "Viewer", "WRITE", "NORMAL"
    )
    assert decision == "BLOCK"
    assert "Insufficient Role Permission" in reasons


def test_lstm_anomaly():
    decision, reasons = make_security_decision(
        "Tenant_A", "Tenant_A",
        "Admin", "READ", "ANOMALY"
    )
    assert decision == "BLOCK"
    assert "LSTM Detected Suspicious Activity" in reasons


def test_lstm_unavailable():
    decision, reasons = make_security_decision(
        "Tenant_A", "Tenant_A",
        "Admin", "READ", "UNAVAILABLE"
    )
    assert decision == "BLOCK"
    assert "LSTM Prediction Unavailable" in reasons


def test_invalid_lstm_status():
    decision, reasons = make_security_decision(
        "Tenant_A", "Tenant_A",
        "Admin", "READ", "UNKNOWN"
    )
    assert decision == "BLOCK"
    assert "Invalid LSTM Status" in reasons
