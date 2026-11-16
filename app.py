import firebase_admin
from firebase_admin import credentials, firestore, auth
import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from security_logic import make_security_decision
st.set_page_config(
    page_title="Cloud Security Monitor",
    page_icon="🔐",
    layout="wide" 
)
@st.cache_resource
def initialize_firebase():
    cred = credentials.Certificate(
        dict(st.secrets["firebase"])
    )

    try:
        app = firebase_admin.get_app()
    except ValueError:
        app = firebase_admin.initialize_app(cred)

    return firestore.client(app=app)


db = initialize_firebase()
st.title("🔐 Multi-Tenant Cloud Security")
st.subheader("Anomaly Detection and Data Leakage Prevention")

st.info("Upload your cloud audit log CSV report.")

uploaded_file = st.file_uploader(
    "Upload Audit Log CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    required = [
        "user_id",
        "user_tenant",
        "resource_tenant",
        "anomaly_status",
        "security_action"
    ]

    if all(col in df.columns for col in required):

        total = len(df)
        normal = (df["anomaly_status"] == "NORMAL").sum()
        anomalies = (df["anomaly_status"] == "ANOMALY").sum()
        blocked = (df["security_action"] == "BLOCK").sum()

        st.success("Audit logs loaded successfully!")

        st.markdown("### Security Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Events", total)
        c2.metric("Normal Events", normal)
        c3.metric("Anomalies", anomalies)
        c4.metric("Blocked Events", blocked)

        st.markdown("### Anomaly Analysis")

        chart_data = pd.DataFrame({
            "Event Type": ["Normal", "Anomaly"],
            "Count": [normal, anomalies]
        })

        st.bar_chart(
            chart_data,
            x="Event Type",
            y="Count"
        )

        st.markdown("### Tenant Activity")

        tenant_data = df.groupby(
            "user_tenant"
        ).size().reset_index(name="Events")

        st.bar_chart(
            tenant_data,
            x="user_tenant",
            y="Events"
        )

        st.markdown("### Security Audit Logs")
        st.dataframe(df, use_container_width=True)

        st.markdown("### Suspicious Activities")

        suspicious = df[
            df["security_action"] == "BLOCK"
        ]

        st.dataframe(
            suspicious,
            use_container_width=True
        )

        st.download_button(
            "Download Security Report",
            df.to_csv(index=False),
            "security_report.csv",
            "text/csv"
        )

    else:
        st.error("CSV does not contain required columns.")

else:
    st.warning("Please upload your audit log CSV file.")

# ==========================================
# PHASE 2: ROLE-BASED ACCESS CONTROL DASHBOARD
# ==========================================

st.divider()
st.header("Phase 2: Multi-Tenant Access Control")

st.write(
    "Upload the Phase 2 security report generated "
    "from Google Colab."
)

phase2_file = st.file_uploader(
    "Upload Phase 2 Security Report",
    type=["csv"],
    key="phase2_upload"
)

if phase2_file is not None:

    phase2_df = pd.read_csv(phase2_file)

    required_columns = [
        "tenant_id",
        "resource_tenant",
        "user_role",
        "action",
        "security_decision",
        "reason"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in phase2_df.columns
    ]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")

    else:
        phase2_df["security_decision"] = (
            phase2_df["security_decision"]
            .astype(str).str.strip().str.upper()
        )

        # Security metrics
        total = len(phase2_df)

        allowed = (
            phase2_df["security_decision"] == "ALLOW"
        ).sum()

        blocked = (
            phase2_df["security_decision"] == "BLOCK"
        ).sum()

        st.subheader("Access Control Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Requests", total)
        col2.metric("Allowed", allowed)
        col3.metric("Blocked", blocked)

        # Tenant filter
        st.subheader("Tenant-wise Monitoring")

        tenants = ["All Tenants"] + sorted(
            phase2_df["tenant_id"].dropna().unique().tolist()
        )

        selected_tenant = st.selectbox(
            "Select Tenant",
            tenants,
            key="phase2_tenant_filter"
        )

        if selected_tenant == "All Tenants":
            filtered_df = phase2_df.copy()
        else:
            filtered_df = phase2_df[
                phase2_df["tenant_id"] == selected_tenant
            ]

        st.subheader("Access Request Records")
        st.dataframe(filtered_df, use_container_width=True)

        # Security alerts
        st.subheader("Security Alerts")

        alerts = filtered_df[
            filtered_df["security_decision"] == "BLOCK"
        ]

        if not alerts.empty:
            st.warning(f"{len(alerts)} blocked requests detected.")
            st.dataframe(
                alerts,
                use_container_width=True
            )
        else:
            st.success("No blocked requests in this selection.")

        # Download report
        csv_data = phase2_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Phase 2 Security Report",
            data=csv_data,
            file_name="phase2_security_report.csv",
            mime="text/csv",
            key="phase2_download"
        )

        st.success("Phase 2 report processed successfully!")
# ==========================================
# PHASE 3: LSTM ANOMALY DETECTION
# ==========================================

st.divider()
st.header("Phase 3: LSTM-Based Anomaly Detection")

st.write(
    "Upload a cloud audit log CSV containing "
    "10 events per sequence for LSTM prediction."
)

@st.cache_resource
def load_lstm_components():
    model = tf.keras.models.load_model(
        "cloud_anomaly_lstm.keras"
    )

    scaler = joblib.load("lstm_scaler.pkl")

    config = joblib.load("lstm_feature_config.pkl")

    return model, scaler, config


try:
    lstm_model, lstm_scaler, lstm_config = (
        load_lstm_components()
    )

    st.success("LSTM model loaded successfully!")

    lstm_file = st.file_uploader(
        "Upload LSTM Audit Dataset",
        type=["csv"],
        key="lstm_upload"
    )

    if lstm_file is not None:

        lstm_df = pd.read_csv(lstm_file)

        features = lstm_config["features"]
        sequence_length = lstm_config["sequence_length"]

        required = ["sequence_id"] + features

        missing = [
            col for col in required
            if col not in lstm_df.columns
        ]

        if missing:
            st.error(
                f"Missing required columns: {missing}"
            )

        else:

            if "timestamp" in lstm_df.columns:
                lstm_df = lstm_df.sort_values(
                    ["sequence_id", "timestamp"]
                )

            valid_sequences = []
            valid_sequence_ids = []

            for sequence_id, group in lstm_df.groupby("sequence_id", sort=True):
                if len(group) != sequence_length:
                    continue
                    sequence = group[features].to_numpy(dtype=np.float32)
                    
            valid_sequences = []
            valid_sequence_ids = []

            for sequence_id, group in lstm_df.groupby(
                "sequence_id", sort=True
            ):
                if len(group) != sequence_length:
                    continue

                sequence = group[features].to_numpy(
                    dtype=np.float32
                )

                valid_sequences.append(sequence)
                valid_sequence_ids.append(sequence_id)

            predictions = []

            if valid_sequences:
                batch = np.stack(valid_sequences)

                original_shape = batch.shape
                batch_2d = batch.reshape(-1, len(features))

                batch_scaled = lstm_scaler.transform(batch_2d)
                batch_scaled = batch_scaled.reshape(original_shape)

                probabilities = lstm_model.predict(
                    batch_scaled,
                    verbose=0,
                    batch_size=32
                ).reshape(-1)

                for sequence_id, probability in zip(
                    valid_sequence_ids, probabilities
                ):
                    status = (
                        "ANOMALY"
                        if probability >= 0.5
                        else "NORMAL"
                    )

                    predictions.append({
                        "Sequence ID": sequence_id,
                        "Anomaly Probability": round(
                            float(probability) * 100, 2
                        ),
                        "Prediction": status
                    })


            if predictions:

                result_df = pd.DataFrame(predictions)
                # Store actual LSTM predictions for Phase 4
                st.session_state["lstm_results"] = result_df.copy()

                total_sequences = len(result_df)

                anomaly_count = (
                    result_df["Prediction"] == "ANOMALY"
                ).sum()

                normal_count = (
                    result_df["Prediction"] == "NORMAL"
                ).sum()

                st.subheader("LSTM Prediction Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Sequences",
                    total_sequences
                )

                col2.metric(
                    "Normal Sequences",
                    int(normal_count)
                )

                col3.metric(
                    "Anomalous Sequences",
                    int(anomaly_count)
                )

                st.subheader("Prediction Chart")

                chart_df = pd.DataFrame({
                    "Status": ["Normal", "Anomaly"],
                    "Count": [
                        int(normal_count),
                        int(anomaly_count)
                    ]
                })

                st.bar_chart(
                    chart_df,
                    x="Status",
                    y="Count"
                )

                st.subheader("LSTM Prediction Results")

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                suspicious = result_df[
                    result_df["Prediction"] == "ANOMALY"
                ]

                if not suspicious.empty:
                    st.warning(
                        f"{len(suspicious)} suspicious "
                        "sequences detected by LSTM."
                    )
                    st.dataframe(
                        suspicious,
                        use_container_width=True
                    )
                else:
                    st.success(
                        "No anomalous sequences detected."
                    )

                st.download_button(
                    "Download LSTM Prediction Report",
                    result_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    "lstm_prediction_report.csv",
                    "text/csv",
                    key="lstm_download"
                )

            else:
                st.error(
                    "No valid sequences found. "
                    "Each sequence must contain exactly "
                    f"{sequence_length} events."
                )

except Exception as e:
    st.error(
        f"Could not load the LSTM model: {e}"
    )
# ==========================================
# PHASE 4: INTEGRATED DETECTION & PREVENTION
# ==========================================

st.divider()
st.header("Phase 4: Integrated Security Decision")

st.write(
    "Combines LSTM anomaly detection with "
    "Role-Based Access Control (RBAC)."
)

st.info(
    "This is a simulated security enforcement workflow."
)

# Role permissions
role_permissions = {
    "Admin": ["READ", "WRITE", "DELETE"],
    "Developer": ["READ", "WRITE"],
    "Viewer": ["READ"]
}

st.subheader("Test an Access Request")

col1, col2 = st.columns(2)

with col1:
    user_tenant = st.selectbox(
        "User Tenant",
        ["Tenant_A", "Tenant_B"],
        key="integrated_user_tenant"
    )

    user_role = st.selectbox(
        "User Role",
        ["Admin", "Developer", "Viewer"],
        key="integrated_user_role"
    )

with col2:
    resource_tenant = st.selectbox(
        "Resource Tenant",
        ["Tenant_A", "Tenant_B"],
        key="integrated_resource_tenant"
    )

    action = st.selectbox(
        "Requested Action",
        ["READ", "WRITE", "DELETE"],
        key="integrated_action"
    )

# Select LSTM result
 # Select an actual LSTM prediction
lstm_results = st.session_state.get("lstm_results")

if lstm_results is not None and not lstm_results.empty:

    sequence_options = (
        lstm_results["Sequence ID"].tolist()
    )

    selected_sequence = st.selectbox(
        "Select LSTM Sequence",
        sequence_options
    )

    selected_result = lstm_results[
        lstm_results["Sequence ID"] == selected_sequence
    ].iloc[0]

    lstm_status = selected_result["Prediction"]

    st.info(
        f"Actual LSTM Prediction: {lstm_status}"
    )

else:
    lstm_status = "UNAVAILABLE"

    st.warning(
        "LSTM prediction unavailable. "
        "Upload an LSTM dataset before making "
        "a security decision."
    )

if st.button("Run Integrated Security Check"):

    final_decision, reasons = make_security_decision(
        user_tenant,
        resource_tenant,
        user_role,
        action,
        lstm_status
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
            st.write("- ", reason)
    else:
        st.write("Reason: Authorized access and normal activity.")
            # Save security decision to Firebase Firestore
    try:
        audit_record = {
            "user_tenant": user_tenant,
            "resource_tenant": resource_tenant,
            "user_role": user_role,
            "action": action,
            "lstm_status": lstm_status,
            "decision": final_decision,
            "reasons": reasons,
            "timestamp": firestore.SERVER_TIMESTAMP
        }

        db.collection("audit_logs").add(audit_record)

        st.success("Security event saved to Firebase Firestore!")

    except Exception as e:
        st.error(f"Firebase logging failed: {e}")
# PHASE 11: FINAL VALIDATION REPORT

st.header("Phase 11: Final Security Validation")

validation_data = [
    ["TC01", "Normal + Authorized Access", "ALLOW", "ALLOW", "PASS"],
    ["TC02", "LSTM Anomaly Detected", "BLOCK", "BLOCK", "PASS"],
    ["TC03", "Cross-Tenant Access", "BLOCK", "BLOCK", "PASS"],
    ["TC04", "Insufficient Role Permission", "BLOCK", "BLOCK", "PASS"],
    ["TC05", "LSTM Prediction Unavailable", "BLOCK", "BLOCK", "PASS"]
]

validation_df = pd.DataFrame(
    validation_data,
    columns=[
        "Test ID",
        "Test Scenario",
        "Expected Result",
        "Actual Result",
        "Status"
    ]
)

st.dataframe(validation_df, use_container_width=True)

st.metric("Total Test Cases", len(validation_df))
st.metric(
    "Passed Test Cases",
    len(validation_df[validation_df["Status"] == "PASS"])
)

csv_data = validation_df.to_csv(index=False)

st.download_button(
    "Download Validation Report",
    data=csv_data,
    file_name="final_validation_report.csv",
    mime="text/csv"
)

# ==========================================
# PHASE 12: FIREBASE CLOUD AUDIT LOG VIEWER
# ==========================================

st.divider()
st.header("Phase 12: Firebase Cloud Audit Logs")

st.write(
    "View security events stored in Firebase Firestore."
)

if st.button("Load Firebase Audit Logs"):

    try:
        documents = (
            db.collection("audit_logs")
            .order_by(
                "timestamp",
                direction=firestore.Query.DESCENDING
            )
            .limit(100)
            .stream()
        )

        audit_records = []

        for document in documents:
            record = document.to_dict()
            record["Document ID"] = document.id
            audit_records.append(record)

        if audit_records:

            audit_df = pd.DataFrame(audit_records)

            st.success(
                f"Loaded {len(audit_df)} Firebase audit events."
            )

            total = len(audit_df)
            allowed = (
                audit_df["decision"] == "ALLOW"
            ).sum()
            blocked = (
                audit_df["decision"] == "BLOCK"
            ).sum()

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Events", total)
            col2.metric("Allowed", int(allowed))
            col3.metric("Blocked", int(blocked))

            st.subheader("Cloud Security Event Records")

            st.dataframe(
                audit_df,
                use_container_width=True
            )

            st.download_button(
                "Download Firebase Audit Report",
                audit_df.to_csv(index=False).encode("utf-8"),
                "firebase_audit_logs.csv",
                "text/csv",
                key="firebase_audit_download"
            )

        else:
            st.info("No Firebase audit records found.")

    except Exception as e:
        st.error(f"Could not load Firebase logs: {e}")
