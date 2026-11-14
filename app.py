import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Cloud Security Monitor",
    page_icon="🔐",
    layout="wide"
)

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
