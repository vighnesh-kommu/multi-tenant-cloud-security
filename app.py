"""Multi-Tenant Cloud Security Monitoring Application."""

import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import firebase_admin

from pathlib import Path
from datetime import datetime, timezone

from firebase_admin import auth, credentials, firestore
from security_logic import make_security_decision


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Cloud Security Monitor",
    page_icon="🔐",
    layout="wide",
)

st.title("🔐 Multi-Tenant Cloud Security")
st.subheader("Anomaly Detection and Data Leakage Prevention")

TENANTS = ["Tenant_A", "Tenant_B"]
ROLES = ["Admin", "Developer", "Viewer"]

MODEL_PATH = Path("cloud_anomaly_lstm.keras")
SCALER_PATH = Path("lstm_scaler.pkl")
CONFIG_PATH = Path("lstm_feature_config.pkl")

LSTM_THRESHOLD = 0.5


# --------------------------------------------------
# 2. FIREBASE INITIALIZATION
# --------------------------------------------------

@st.cache_resource
def initialize_firebase():
    """Initialize Firebase Admin and return Firestore client."""

    firebase_data = dict(st.secrets["firebase"])
    firebase_credentials = credentials.Certificate(firebase_data)

    try:
        firebase_app = firebase_admin.get_app()
    except ValueError:
        firebase_app = firebase_admin.initialize_app(
            firebase_credentials
        )

    return firestore.client(app=firebase_app)


try:
    db = initialize_firebase()
except Exception as error:
    st.error("Firebase initialization failed.")
    st.caption(str(error))
    st.stop()


# --------------------------------------------------
# 3. FIREBASE AUTHENTICATION
# --------------------------------------------------

def firebase_login(email, password):
    """Authenticate a user using Firebase Email/Password."""

    api_key = st.secrets["firebase_web"]["api_key"]

    url = (
        "https://identitytoolkit.googleapis.com/v1/"
        f"accounts:signInWithPassword?key={api_key}"
    )

    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True,
    }

    response = requests.post(
        url,
        json=payload,
        timeout=15,
    )

    result = response.json()

    if response.status_code != 200:
        error_message = (
            result.get("error", {}).get("message", "Login failed")
        )
        raise ValueError(error_message)

    return result["idToken"]


def get_authenticated_profile(id_token):
    """Verify the Firebase token and load the user's tenant profile."""

    decoded_token = auth.verify_id_token(
        id_token,
        check_revoked=True,
    )

    user_uid = decoded_token["uid"]

    profile_document = (
        db.collection("user_tenants")
        .document(user_uid)
        .get()
    )

    if not profile_document.exists:
        raise ValueError("User tenant profile was not found.")

    profile = profile_document.to_dict()

    tenant_id = profile.get("tenant_id")
    role = profile.get("role")

    if tenant_id not in TENANTS:
        raise ValueError("Invalid tenant configuration.")

    if role not in ROLES:
        raise ValueError("Invalid user role configuration.")

    return {
        "uid": user_uid,
        "email": decoded_token.get("email", ""),
        "tenant_id": tenant_id,
        "role": role,
    }


def clear_login_session():
    """Clear authentication details from the current session."""

    for key in ("id_token", "user_profile", "lstm_results"):
        st.session_state.pop(key, None)


def show_login_page():
    """Display the Firebase login form."""

    st.title("🔑 Secure User Login")
    st.write("Sign in using your registered Firebase account.")

    with st.form("login_form"):
        email = st.text_input("Email address")
        password = st.text_input(
            "Password",
            type="password",
        )

        submitted = st.form_submit_button(
            "Login",
            use_container_width=True,
        )

    if submitted:
        if not email.strip() or not password:
            st.warning("Enter both email and password.")
            return

        try:
            token = firebase_login(email.strip(), password)
            profile = get_authenticated_profile(token)

            st.session_state["id_token"] = token
            st.session_state["user_profile"] = profile

            st.success("Login successful.")
            st.rerun()

        except Exception:
            st.error(
                "Login failed. Check your credentials and "
                "Firebase user profile."
            )


def require_authenticated_user():
    """Verify the current session before displaying protected pages."""

    token = st.session_state.get("id_token")

    if not token:
        show_login_page()
        st.stop()

    try:
        profile = get_authenticated_profile(token)
        st.session_state["user_profile"] = profile
        return profile

    except Exception:
        clear_login_session()
        st.warning("Your session has expired. Please log in again.")
        st.rerun()


# --------------------------------------------------
# 4. TENANT DATA FILTERING
# --------------------------------------------------

def filter_tenant_data(dataframe, tenant_column, tenant_id):
    """Return only records belonging to the authenticated tenant."""

    if tenant_column not in dataframe.columns:
        raise ValueError(
            f"Required tenant column '{tenant_column}' is missing."
        )

    return dataframe[
        dataframe[tenant_column].astype(str).str.strip() == tenant_id
    ].copy()


def display_csv_download(dataframe, filename, label):
    """Display a CSV download button."""

    csv_data = dataframe.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


# --------------------------------------------------
# 5. AUTHENTICATED DASHBOARD
# --------------------------------------------------

profile = require_authenticated_user()

user_uid = profile["uid"]
user_email = profile["email"]
user_tenant = profile["tenant_id"]
user_role = profile["role"]

with st.sidebar:
    st.header("👤 Account Details")
    st.write(f"**Email:** {user_email}")
    st.write(f"**Tenant:** {user_tenant}")
    st.write(f"**Role:** {user_role}")

    if st.button("Logout", use_container_width=True):
        clear_login_session()
        st.rerun()


st.success(f"Authenticated tenant: {user_tenant}")

st.divider()


# --------------------------------------------------
# 6. CLOUD AUDIT LOG DASHBOARD
# --------------------------------------------------

st.header("📊 Cloud Audit Log Dashboard")

st.info(
    "Upload an audit log CSV containing the required tenant "
    "and security columns."
)

uploaded_file = st.file_uploader(
    "Upload Audit Log CSV",
    type=["csv"],
    key="dashboard_csv",
)

if uploaded_file is not None:
    try:
        audit_df = pd.read_csv(uploaded_file)

        required_columns = {
            "user_id",
            "user_tenant",
            "resource_tenant",
            "anomaly_status",
            "security_action",
        }

        missing_columns = required_columns - set(audit_df.columns)

        if missing_columns:
            st.error(
                "Missing columns: "
                + ", ".join(sorted(missing_columns))
            )
        else:
            # Never show another tenant's dashboard records.
            audit_df = filter_tenant_data(
                audit_df,
                "user_tenant",
                user_tenant,
            )

            if audit_df.empty:
                st.warning(
                    "No audit records were found for your tenant."
                )
            else:
                total_records = len(audit_df)

                anomaly_count = (
                    audit_df["anomaly_status"]
                    .astype(str)
                    .str.upper()
                    .eq("ANOMALY")
                    .sum()
                )

                blocked_count = (
                    audit_df["security_action"]
                    .astype(str)
                    .str.upper()
                    .eq("BLOCK")
                    .sum()
                )

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Events", total_records)
                col2.metric("Anomalies", int(anomaly_count))
                col3.metric("Blocked Events", int(blocked_count))

                st.subheader("Audit Log Records")
                st.dataframe(
                    audit_df,
                    use_container_width=True,
                )

                suspicious_df = audit_df[
                    audit_df["security_action"]
                    .astype(str)
                    .str.upper()
                    .eq("BLOCK")
                ]

                st.subheader("🚨 Suspicious Events")
                st.dataframe(
                    suspicious_df,
                    use_container_width=True,
                )

                display_csv_download(
                    audit_df,
                    "tenant_audit_report.csv",
                    "Download Tenant Audit Report",
                )

    except Exception as error:
        st.error("Unable to process the uploaded audit log.")
        st.caption(str(error))


st.divider()


# --------------------------------------------------
# 7. LSTM ANOMALY DETECTION
# --------------------------------------------------

st.header("🤖 LSTM Anomaly Detection")

st.write(
    "Upload sequential audit logs to identify unusual activity "
    "using the trained LSTM model."
)


@st.cache_resource
def load_lstm_assets():
    """Load the trained model, scaler and feature configuration."""

    if not all(
        path.exists()
        for path in (MODEL_PATH, SCALER_PATH, CONFIG_PATH)
    ):
        raise FileNotFoundError(
            "One or more LSTM model files are missing."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    config = joblib.load(CONFIG_PATH)

    features = config["features"]
    sequence_length = int(config["sequence_length"])

    return model, scaler, features, sequence_length


lstm_file = st.file_uploader(
    "Upload LSTM Audit Log CSV",
    type=["csv"],
    key="lstm_csv",
)

if lstm_file is not None:
    try:
        model, scaler, features, sequence_length = load_lstm_assets()

        lstm_df = pd.read_csv(lstm_file)

        required_columns = {"sequence_id", *features}
        missing_columns = required_columns - set(lstm_df.columns)

        if missing_columns:
            st.error(
                "Missing LSTM columns: "
                + ", ".join(sorted(missing_columns))
            )

        else:
            # If tenant metadata exists, enforce tenant filtering.
            tenant_column = None

            if "user_tenant" in lstm_df.columns:
                tenant_column = "user_tenant"
            elif "tenant_id" in lstm_df.columns:
                tenant_column = "tenant_id"

            if tenant_column:
                lstm_df = filter_tenant_data(
                    lstm_df,
                    tenant_column,
                    user_tenant,
                )
            else:
                st.warning(
                    "This CSV has no tenant column. It will be "
                    "treated as a pre-filtered tenant-specific file. "
                    "Its tenant ownership cannot be verified."
                )

            if "timestamp" in lstm_df.columns:
                lstm_df = lstm_df.sort_values("timestamp")

            valid_sequences = []
            valid_sequence_ids = []

            for sequence_id, group in lstm_df.groupby(
                "sequence_id",
                sort=True,
            ):
                if len(group) != sequence_length:
                    continue

                try:
                    sequence = group[features].apply(
                        pd.to_numeric,
                        errors="coerce",
                    ).to_numpy(dtype=np.float32)

                    if not np.isfinite(sequence).all():
                        continue

                    valid_sequences.append(sequence)
                    valid_sequence_ids.append(sequence_id)

                except (TypeError, ValueError):
                    continue

            if not valid_sequences:
                st.warning(
                    "No valid sequences were found. Each sequence "
                    f"must contain exactly {sequence_length} valid rows."
                )
                st.session_state.pop("lstm_results", None)

            else:
                sequences = np.asarray(
                    valid_sequences,
                    dtype=np.float32,
                )

                original_shape = sequences.shape

                scaled_sequences = scaler.transform(
                    sequences.reshape(-1, len(features))
                )

                scaled_sequences = scaled_sequences.reshape(
                    original_shape
                )

                predictions = model.predict(
                    scaled_sequences,
                    verbose=0,
                )

                scores = np.asarray(predictions).reshape(-1)

                results = pd.DataFrame({
                    "sequence_id": valid_sequence_ids,
                    "anomaly_score": scores,
                })

                results["status"] = np.where(
                    results["anomaly_score"] >= LSTM_THRESHOLD,
                    "ANOMALY",
                    "NORMAL",
                )

                st.session_state["lstm_results"] = results

                st.success(
                    f"Processed {len(results)} valid sequences."
                )

                st.dataframe(
                    results,
                    use_container_width=True,
                )

                anomaly_total = int(
                    results["status"].eq("ANOMALY").sum()
                )

                normal_total = int(
                    results["status"].eq("NORMAL").sum()
                )

                col1, col2 = st.columns(2)
                col1.metric("Normal Sequences", normal_total)
                col2.metric("Anomalous Sequences", anomaly_total)

                display_csv_download(
                    results,
                    "lstm_prediction_results.csv",
                    "Download LSTM Results",
                )

    except FileNotFoundError as error:
        st.error(str(error))

    except Exception as error:
        st.error("LSTM processing failed.")
        st.caption(str(error))


st.divider()


# --------------------------------------------------
# 8. INTEGRATED SECURITY DECISION
# --------------------------------------------------

st.header("🛡️ Integrated Security Decision")

st.write(
    "The security engine combines tenant isolation, role-based "
    "permissions and LSTM anomaly detection."
)

st.text_input(
    "Authenticated User Tenant",
    value=user_tenant,
    disabled=True,
)

st.text_input(
    "Authenticated User Role",
    value=user_role,
    disabled=True,
)

resource_tenant = st.selectbox(
    "Select Resource Tenant",
    TENANTS,
)

action = st.selectbox(
    "Select Requested Action",
    ["READ", "WRITE", "DELETE"],
)

lstm_results = st.session_state.get("lstm_results")

if lstm_results is not None and not lstm_results.empty:
    sequence_options = lstm_results["sequence_id"].tolist()

    selected_sequence = st.selectbox(
        "Select LSTM Sequence",
        sequence_options,
    )

    selected_result = lstm_results[
        lstm_results["sequence_id"] == selected_sequence
    ]

    lstm_status = selected_result.iloc[0]["status"]

else:
    st.warning(
        "No LSTM prediction is currently available. "
        "The security engine will fail safely and BLOCK the request."
    )
    lstm_status = "UNAVAILABLE"


if st.button(
    "Run Integrated Security Check",
    use_container_width=True,
):
    final_decision, reasons = make_security_decision(
        user_tenant=user_tenant,
        resource_tenant=resource_tenant,
        user_role=user_role,
        action=action,
        lstm_status=lstm_status,
    )

    if final_decision == "ALLOW":
        st.success("✅ ACCESS ALLOWED")
    else:
        st.error("🚨 ACCESS BLOCKED")

    st.write(f"**LSTM Status:** {lstm_status}")
    st.write(f"**Final Decision:** {final_decision}")

    if reasons:
        st.subheader("Security Reasons")

        for reason in reasons:
            st.warning(reason)
    else:
        st.info("All security checks passed.")

    audit_record = {
        "user_uid": user_uid,
        "user_email": user_email,
        "user_tenant": user_tenant,
        "resource_tenant": resource_tenant,
        "user_role": user_role,
        "action": action,
        "lstm_status": lstm_status,
        "decision": final_decision,
        "reasons": reasons,
        "timestamp": firestore.SERVER_TIMESTAMP,
    }

    try:
        db.collection("audit_logs").add(audit_record)
        st.success("Security event saved to Firestore.")

    except Exception:
        st.error("The security event could not be saved to Firestore.")


st.divider()


# --------------------------------------------------
# 9. SECURITY TEST SCENARIOS
# --------------------------------------------------

st.header("🧪 Security Test Scenarios")

st.caption(
    "These are expected test scenarios, not live test results. "
    "Actual test outcomes are determined by the automated test suite."
)

test_scenarios = pd.DataFrame([
    {
        "Scenario": "Same-tenant READ with Admin",
        "Expected Decision": "ALLOW",
        "Expected Reason": "No security violation",
    },
    {
        "Scenario": "Cross-tenant access",
        "Expected Decision": "BLOCK",
        "Expected Reason": "Cross-Tenant Access Attempt",
    },
    {
        "Scenario": "Viewer requests WRITE",
        "Expected Decision": "BLOCK",
        "Expected Reason": "Insufficient Role Permission",
    },
    {
        "Scenario": "LSTM detects anomaly",
        "Expected Decision": "BLOCK",
        "Expected Reason": "LSTM Detected Suspicious Activity",
    },
    {
        "Scenario": "LSTM prediction unavailable",
        "Expected Decision": "BLOCK",
        "Expected Reason": "LSTM Prediction Unavailable",
    },
])

st.dataframe(
    test_scenarios,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# --------------------------------------------------
# 10. FIRESTORE AUDIT LOG VIEWER
# --------------------------------------------------

st.header("📁 Firestore Security Audit Logs")

st.write(
    f"Showing security events belonging to **{user_tenant}** only."
)

if st.button(
    "Refresh Audit Logs",
    use_container_width=True,
):
    st.session_state["refresh_audit_logs"] = True


try:
    audit_documents = (
        db.collection("audit_logs")
        .where("user_tenant", "==", user_tenant)
        .order_by(
            "timestamp",
            direction=firestore.Query.DESCENDING,
        )
        .limit(100)
        .stream()
    )

    firestore_records = []

    for document in audit_documents:
        record = document.to_dict()
        record["document_id"] = document.id
        firestore_records.append(record)

    if firestore_records:
        firestore_df = pd.DataFrame(firestore_records)

        total_events = len(firestore_df)

        allowed_events = (
            firestore_df["decision"]
            .astype(str)
            .str.upper()
            .eq("ALLOW")
            .sum()
        )

        blocked_events = (
            firestore_df["decision"]
            .astype(str)
            .str.upper()
            .eq("BLOCK")
            .sum()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Events", total_events)
        col2.metric("Allowed", int(allowed_events))
        col3.metric("Blocked", int(blocked_events))

        st.dataframe(
            firestore_df,
            use_container_width=True,
            hide_index=True,
        )

        display_csv_download(
            firestore_df,
            "firestore_tenant_audit_logs.csv",
            "Download Firestore Audit Logs",
        )

    else:
        st.info("No audit logs found for this tenant.")

except Exception as error:
    st.error("Unable to retrieve Firestore audit logs.")
    st.caption(
        "If Firestore reports a missing index, follow the "
        "index creation link in the error message."
    )
    st.caption(str(error))


st.caption(
    "Multi-Tenant Cloud Data Isolation and Leakage Prevention "
    "using Cloud Audit Log Analysis."
                                           )
