"""
Multi-Tenant Cloud Security Monitor

A Streamlit dashboard that combines:
  1. Firebase login with tenant + role lookup
  2. Audit-log and access-control report viewers
  3. LSTM-based anomaly detection on audit-log sequences
  4. An integrated decision (LSTM result + tenant check + RBAC)
  5. A validation report and a Firestore audit-log viewer
"""

import firebase_admin
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf
from firebase_admin import auth, credentials, firestore

from security_logic import ROLE_PERMISSIONS, make_security_decision

st.set_page_config(
    page_title="Cloud Security Monitor",
    page_icon="🔐",
    layout="wide",
)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
VALID_TENANTS = ["Tenant_A", "Tenant_B"]
VALID_ROLES = list(ROLE_PERMISSIONS)
ACTIONS = ["READ", "WRITE", "DELETE"]

ANOMALY_THRESHOLD = 0.5   # probability at or above this is flagged as ANOMALY
LSTM_BATCH_SIZE = 32

AUDIT_LOG_COLUMNS = [
    "user_id",
    "user_tenant",
    "resource_tenant",
    "anomaly_status",
    "security_action",
]
ACCESS_REPORT_COLUMNS = [
    "tenant_id",
    "resource_tenant",
    "user_role",
    "action",
    "security_decision",
    "reason",
]


# ------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------
def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def find_missing_columns(df, required):
    return [col for col in required if col not in df.columns]


# ------------------------------------------------------------------
# Firebase and authentication
# ------------------------------------------------------------------
@st.cache_resource
def initialize_firebase():
    """Start Firebase once and reuse the Firestore client."""
    try:
        app = firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(dict(st.secrets["firebase"]))
        app = firebase_admin.initialize_app(cred)

    return firestore.client(app=app)


def firebase_login(email, password):
    """Sign in through the Firebase REST API and return the ID token."""
    api_key = st.secrets["firebase_web"]["api_key"]
    url = (
        "https://identitytoolkit.googleapis.com/v1/"
        f"accounts:signInWithPassword?key={api_key}"
    )

    response = requests.post(
        url,
        json={"email": email, "password": password, "returnSecureToken": True},
        timeout=15,
    )

    if response.status_code != 200:
        try:
            message = response.json().get("error", {}).get("message", "Login failed")
        except ValueError:
            message = "Login failed"
        raise ValueError(message)

    return response.json()["idToken"]


def get_authenticated_profile(db, id_token):
    """Verify the token and load the user's tenant and role from Firestore."""
    decoded_token = auth.verify_id_token(id_token)
    uid = decoded_token["uid"]

    user_doc = db.collection("user_tenants").document(uid).get()
    if not user_doc.exists:
        raise ValueError("No tenant mapping found for this user.")

    profile = user_doc.to_dict()
    tenant_id = profile.get("tenant_id")
    role = profile.get("role")

    if tenant_id not in VALID_TENANTS:
        raise ValueError("Invalid tenant mapping.")
    if role not in VALID_ROLES:
        raise ValueError("Invalid user role.")

    return {
        "uid": uid,
        "email": decoded_token.get("email"),
        "tenant_id": tenant_id,
        "role": role,
    }


def show_login_page(db):
    st.title("🔐 Multi-Tenant Cloud Security")
    st.subheader("Firebase Login")

    notice = st.session_state.pop("login_notice", None)
    if notice:
        st.warning(notice)

    with st.form("firebase_login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        login_clicked = st.form_submit_button("Login")

    if login_clicked:
        try:
            token = firebase_login(email, password)
            profile = get_authenticated_profile(db, token)
        except Exception as e:
            st.error(f"Login failed: {e}")
            return

        st.session_state["firebase_id_token"] = token
        st.session_state["user_profile"] = profile
        st.rerun()


def log_out():
    st.session_state["firebase_id_token"] = None
    st.session_state["user_profile"] = None
    st.rerun()


def require_login(db):
    """Stop the script until someone is logged in; return their profile."""
    st.session_state.setdefault("firebase_id_token", None)
    st.session_state.setdefault("user_profile", None)

    token = st.session_state["firebase_id_token"]
    if not token:
        show_login_page(db)
        st.stop()

    # Re-check on every run so expired or revoked sessions are caught.
    try:
        profile = get_authenticated_profile(db, token)
    except Exception:
        st.session_state["firebase_id_token"] = None
        st.session_state["user_profile"] = None
        st.session_state["login_notice"] = "Session expired. Please log in again."
        st.rerun()

    st.session_state["user_profile"] = profile
    return profile


def show_sidebar(profile):
    st.sidebar.success(
        f"Logged in: {profile.get('email')}\n"
        f"Tenant: {profile['tenant_id']}\n"
        f"Role: {profile['role']}"
    )
    if st.sidebar.button("Logout"):
        log_out()


# ------------------------------------------------------------------
# Access decision (shared by Phase 4 and the Phase 11 test cases)
# ------------------------------------------------------------------
def evaluate_access_request(user_tenant, resource_tenant, role, action, lstm_status):
    """Tenant, role and LSTM checks all live in security_logic."""
    return make_security_decision(
        user_tenant, resource_tenant, role, action, lstm_status
    )


# ------------------------------------------------------------------
# Phase 1: audit-log overview
# ------------------------------------------------------------------
def render_audit_overview():
    st.title("Multi-Tenant Cloud Security")
    st.subheader("Anomaly Detection and Data Leakage Prevention")
    st.info("Upload your cloud audit log CSV report.")

    uploaded_file = st.file_uploader("Upload Audit Log CSV", type=["csv"])
    if uploaded_file is None:
        st.warning("Please upload your audit log CSV file.")
        return

    df = pd.read_csv(uploaded_file)
    if find_missing_columns(df, AUDIT_LOG_COLUMNS):
        st.error("CSV does not contain required columns.")
        return

    total = len(df)
    normal = int((df["anomaly_status"] == "NORMAL").sum())
    anomalies = int((df["anomaly_status"] == "ANOMALY").sum())
    blocked = int((df["security_action"] == "BLOCK").sum())

    st.success("Audit logs loaded successfully!")

    st.markdown("### Security Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events", total)
    c2.metric("Normal Events", normal)
    c3.metric("Anomalies", anomalies)
    c4.metric("Blocked Events", blocked)

    st.markdown("### Anomaly Analysis")
    anomaly_chart = pd.DataFrame(
        {"Event Type": ["Normal", "Anomaly"], "Count": [normal, anomalies]}
    )
    st.bar_chart(anomaly_chart, x="Event Type", y="Count")

    st.markdown("### Tenant Activity")
    events_per_tenant = df.groupby("user_tenant").size().reset_index(name="Events")
    st.bar_chart(events_per_tenant, x="user_tenant", y="Events")

    st.markdown("### Security Audit Logs")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Suspicious Activities")
    st.dataframe(df[df["security_action"] == "BLOCK"], use_container_width=True)

    st.download_button(
        "Download Security Report",
        to_csv_bytes(df),
        "security_report.csv",
        "text/csv",
    )


# ------------------------------------------------------------------
# Phase 2: role-based access control report
# ------------------------------------------------------------------
def render_access_control_report():
    st.divider()
    st.header("Phase 2: Multi-Tenant Access Control")
    st.write("Upload the Phase 2 security report generated from Google Colab.")

    report_file = st.file_uploader(
        "Upload Phase 2 Security Report", type=["csv"], key="phase2_upload"
    )
    if report_file is None:
        return

    report_df = pd.read_csv(report_file)

    missing = find_missing_columns(report_df, ACCESS_REPORT_COLUMNS)
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    report_df["security_decision"] = (
        report_df["security_decision"].astype(str).str.strip().str.upper()
    )

    total = len(report_df)
    allowed = int((report_df["security_decision"] == "ALLOW").sum())
    blocked = int((report_df["security_decision"] == "BLOCK").sum())

    st.subheader("Access Control Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Requests", total)
    col2.metric("Allowed", allowed)
    col3.metric("Blocked", blocked)

    st.subheader("Tenant-wise Monitoring")
    tenant_options = ["All Tenants"] + sorted(
        report_df["tenant_id"].dropna().unique().tolist()
    )
    selected_tenant = st.selectbox(
        "Select Tenant", tenant_options, key="phase2_tenant_filter"
    )

    if selected_tenant == "All Tenants":
        filtered_df = report_df
    else:
        filtered_df = report_df[report_df["tenant_id"] == selected_tenant]

    st.subheader("Access Request Records")
    st.dataframe(filtered_df, use_container_width=True)

    st.subheader("Security Alerts")
    alerts = filtered_df[filtered_df["security_decision"] == "BLOCK"]
    if alerts.empty:
        st.success("No blocked requests in this selection.")
    else:
        st.warning(f"{len(alerts)} blocked requests detected.")
        st.dataframe(alerts, use_container_width=True)

    st.download_button(
        label="Download Phase 2 Security Report",
        data=to_csv_bytes(report_df),
        file_name="phase2_security_report.csv",
        mime="text/csv",
        key="phase2_download",
    )
    st.success("Phase 2 report processed successfully!")


# ------------------------------------------------------------------
# Phase 3: LSTM anomaly detection
# ------------------------------------------------------------------
@st.cache_resource
def load_lstm_components():
    model = tf.keras.models.load_model("cloud_anomaly_lstm.keras")
    scaler = joblib.load("lstm_scaler.pkl")
    config = joblib.load("lstm_feature_config.pkl")
    return model, scaler, config


def build_sequences(df, features, sequence_length):
    """Group rows by sequence_id and keep only complete sequences."""
    sequence_ids = []
    sequences = []

    for sequence_id, group in df.groupby("sequence_id", sort=True):
        if len(group) != sequence_length:
            continue
        sequence_ids.append(sequence_id)
        sequences.append(group[features].to_numpy(dtype=np.float32))

    return sequence_ids, sequences


def predict_sequences(model, scaler, sequence_ids, sequences, n_features):
    """Scale the sequences, run the model, and return a results table."""
    batch = np.stack(sequences)
    original_shape = batch.shape

    # The scaler expects 2D input, so flatten, scale, then restore the shape.
    scaled = scaler.transform(batch.reshape(-1, n_features)).reshape(original_shape)

    probabilities = model.predict(
        scaled, verbose=0, batch_size=LSTM_BATCH_SIZE
    ).reshape(-1)

    return pd.DataFrame(
        {
            "Sequence ID": sequence_ids,
            "Anomaly Probability": np.round(probabilities.astype(float) * 100, 2),
            "Prediction": np.where(
                probabilities >= ANOMALY_THRESHOLD, "ANOMALY", "NORMAL"
            ),
        }
    )


def render_lstm_detection():
    st.divider()
    st.header("Phase 3: LSTM-Based Anomaly Detection")
    st.write(
        "Upload a cloud audit log CSV containing "
        "10 events per sequence for LSTM prediction."
    )

    try:
        model, scaler, config = load_lstm_components()
    except Exception as e:
        st.error(f"Could not load the LSTM model: {e}")
        return

    st.success("LSTM model loaded successfully!")

    lstm_file = st.file_uploader(
        "Upload LSTM Audit Dataset", type=["csv"], key="lstm_upload"
    )
    if lstm_file is None:
        return

    lstm_df = pd.read_csv(lstm_file)
    features = config["features"]
    sequence_length = config["sequence_length"]

    missing = find_missing_columns(lstm_df, ["sequence_id"] + features)
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    if "timestamp" in lstm_df.columns:
        lstm_df = lstm_df.sort_values(["sequence_id", "timestamp"])

    sequence_ids, sequences = build_sequences(lstm_df, features, sequence_length)
    if not sequences:
        st.error(
            "No valid sequences found. Each sequence must contain "
            f"exactly {sequence_length} events."
        )
        return

    try:
        result_df = predict_sequences(
            model, scaler, sequence_ids, sequences, len(features)
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    # Phase 4 reads these results from the session.
    st.session_state["lstm_results"] = result_df.copy()

    anomaly_count = int((result_df["Prediction"] == "ANOMALY").sum())
    normal_count = int((result_df["Prediction"] == "NORMAL").sum())

    st.subheader("LSTM Prediction Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sequences", len(result_df))
    col2.metric("Normal Sequences", normal_count)
    col3.metric("Anomalous Sequences", anomaly_count)

    st.subheader("Prediction Chart")
    chart_df = pd.DataFrame(
        {"Status": ["Normal", "Anomaly"], "Count": [normal_count, anomaly_count]}
    )
    st.bar_chart(chart_df, x="Status", y="Count")

    st.subheader("LSTM Prediction Results")
    st.dataframe(result_df, use_container_width=True)

    flagged = result_df[result_df["Prediction"] == "ANOMALY"]
    if flagged.empty:
        st.success("No anomalous sequences detected.")
    else:
        st.warning(f"{len(flagged)} suspicious sequences detected by LSTM.")
        st.dataframe(flagged, use_container_width=True)

    st.download_button(
        "Download LSTM Prediction Report",
        to_csv_bytes(result_df),
        "lstm_prediction_report.csv",
        "text/csv",
        key="lstm_download",
    )


# ------------------------------------------------------------------
# Phase 4: integrated detection and prevention
# ------------------------------------------------------------------
def render_integrated_decision(db, profile):
    st.divider()
    st.header("Phase 4: Integrated Security Decision")
    st.write("Combines LSTM anomaly detection with Role-Based Access Control (RBAC).")
    st.info("This is a simulated security enforcement workflow.")

    user_tenant = profile["tenant_id"]
    user_role = profile["role"]

    st.subheader("Test an Access Request")
    col1, col2 = st.columns(2)

    with col1:
        st.text_input("Authenticated User Tenant", value=user_tenant, disabled=True)
        st.text_input("Authenticated User Role", value=user_role, disabled=True)

    with col2:
        resource_tenant = st.selectbox(
            "Resource Tenant", VALID_TENANTS, key="integrated_resource_tenant"
        )
        action = st.selectbox("Requested Action", ACTIONS, key="integrated_action")

    # Use a real LSTM prediction if Phase 3 has produced one.
    lstm_results = st.session_state.get("lstm_results")

    if lstm_results is not None and not lstm_results.empty:
        selected_sequence = st.selectbox(
            "Select LSTM Sequence", lstm_results["Sequence ID"].tolist()
        )
        selected_row = lstm_results[
            lstm_results["Sequence ID"] == selected_sequence
        ].iloc[0]
        lstm_status = selected_row["Prediction"]
        st.info(f"Actual LSTM Prediction: {lstm_status}")
    else:
        lstm_status = "UNAVAILABLE"
        st.warning(
            "LSTM prediction unavailable. Upload an LSTM dataset "
            "before making a security decision."
        )

    if not st.button("Run Integrated Security Check"):
        return

    final_decision, reasons = evaluate_access_request(
        user_tenant, resource_tenant, user_role, action, lstm_status
    )

    if final_decision == "BLOCK":
        st.error("ACCESS BLOCKED")
    else:
        st.success("ACCESS ALLOWED")

    st.subheader("Security Decision Report")
    st.write("User Tenant:", user_tenant)
    st.write("Resource Tenant:", resource_tenant)
    st.write("User Role:", user_role)
    st.write("Requested Action:", action)
    st.write("LSTM Status:", lstm_status)
    st.write("Final Decision:", final_decision)

    if reasons:
        st.write("Security Reasons:")
        for reason in reasons:
            st.markdown(f"- {reason}")
    else:
        st.write("Reason: Authorized access and normal activity.")

    # Save the decision so it shows up in the Phase 12 log viewer.
    try:
        db.collection("audit_logs").add(
            {
                "user_id": profile["uid"],
                "user_tenant": user_tenant,
                "resource_tenant": resource_tenant,
                "user_role": user_role,
                "action": action,
                "lstm_status": lstm_status,
                "decision": final_decision,
                "reasons": reasons,
                "timestamp": firestore.SERVER_TIMESTAMP,
            }
        )
        st.success("Security event saved to Firebase Firestore!")
    except Exception as e:
        st.error(f"Firebase logging failed: {e}")


# ------------------------------------------------------------------
# Phase 11: validation report
# ------------------------------------------------------------------
# (id, scenario, user tenant, resource tenant, role, action, LSTM status, expected)
TEST_CASES = [
    ("TC01", "Normal + Authorized Access", "Tenant_A", "Tenant_A", "Developer", "READ", "NORMAL", "ALLOW"),
    ("TC02", "LSTM Anomaly Detected", "Tenant_A", "Tenant_A", "Developer", "READ", "ANOMALY", "BLOCK"),
    ("TC03", "Cross-Tenant Access", "Tenant_A", "Tenant_B", "Admin", "READ", "NORMAL", "BLOCK"),
    ("TC04", "Insufficient Role Permission", "Tenant_A", "Tenant_A", "Viewer", "DELETE", "NORMAL", "BLOCK"),
    ("TC05", "LSTM Prediction Unavailable", "Tenant_A", "Tenant_A", "Admin", "READ", "UNAVAILABLE", "BLOCK"),
]


def run_validation_tests():
    """Run each test case through the real decision logic."""
    rows = []
    for case_id, scenario, u_tenant, r_tenant, role, action, lstm, expected in TEST_CASES:
        actual, _ = evaluate_access_request(u_tenant, r_tenant, role, action, lstm)
        rows.append(
            {
                "Test ID": case_id,
                "Test Scenario": scenario,
                "Expected Result": expected,
                "Actual Result": actual,
                "Status": "PASS" if actual == expected else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def render_validation_report():
    st.divider()
    st.header("Phase 11: Final Security Validation")

    validation_df = run_validation_tests()
    st.dataframe(validation_df, use_container_width=True)

    st.metric("Total Test Cases", len(validation_df))
    st.metric("Passed Test Cases", int((validation_df["Status"] == "PASS").sum()))

    st.download_button(
        "Download Validation Report",
        data=to_csv_bytes(validation_df),
        file_name="final_validation_report.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------------
# Phase 12: Firebase audit-log viewer
# ------------------------------------------------------------------
def render_audit_log_viewer(db, profile):
    st.divider()
    st.header("Phase 12: Firebase Cloud Audit Logs")
    st.write("View security events stored in Firebase Firestore.")

    if not st.button("Load Firebase Audit Logs"):
        return

    try:
        documents = (
            db.collection("audit_logs")
            .where("user_tenant", "==", profile["tenant_id"])
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(100)
            .stream()
        )

        records = []
        for document in documents:
            record = document.to_dict()
            record["Document ID"] = document.id
            records.append(record)
    except Exception as e:
        st.error(f"Could not load Firebase logs: {e}")
        return

    if not records:
        st.info("No Firebase audit records found.")
        return

    audit_df = pd.DataFrame(records)
    decisions = audit_df.get("decision", pd.Series(dtype=str))

    st.success(f"Loaded {len(audit_df)} Firebase audit events.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Events", len(audit_df))
    col2.metric("Allowed", int((decisions == "ALLOW").sum()))
    col3.metric("Blocked", int((decisions == "BLOCK").sum()))

    st.subheader("Cloud Security Event Records")
    st.dataframe(audit_df, use_container_width=True)

    st.download_button(
        "Download Firebase Audit Report",
        to_csv_bytes(audit_df),
        "firebase_audit_logs.csv",
        "text/csv",
        key="firebase_audit_download",
    )


# ------------------------------------------------------------------
# Page layout
# ------------------------------------------------------------------
db = initialize_firebase()
profile = require_login(db)
show_sidebar(profile)

render_audit_overview()
render_access_control_report()
render_lstm_detection()
render_integrated_decision(db, profile)
render_validation_report()
render_audit_log_viewer(db, profile)
