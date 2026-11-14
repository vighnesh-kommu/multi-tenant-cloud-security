import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Multi-Tenant Cloud Security",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 Multi-Tenant Cloud Security")
st.subheader("Anomaly Detection & Data Leakage Prevention")

st.sidebar.title("Security Controls")

uploaded_file = st.file_uploader(
    "Upload Cloud Audit Log CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    required = [
        "user_id", "user_tenant",
        "resource_tenant", "status",
        "anomaly_status", "security_action"
    ]

    if not all(col in df.columns for col in required):
        st.error("Invalid CSV format. Please upload the corrected report.")

    else:

        # Normalize values
        for col in [
            "anomaly_status", "security_action", "status"
        ]:
            df[col] = df[col].astype(str).str.strip().str.upper()

        # Explain detected events
        def get_reason(row):
            reasons = []

            if row["user_tenant"] != row["resource_tenant"]:
                reasons.append("Cross-Tenant Access")

            if row["status"] == "FAILED":
                reasons.append("Failed Login")

            if row["status"] == "FAILED":
                if row["user_id"] in repeated_users:
                    reasons.append("Repeated Failed Login")

            return ", ".join(reasons) if reasons else "Authorized Activity"

        failed_counts = (
            df[df["status"] == "FAILED"]
            .groupby("user_id").size()
        )

        repeated_users = set(
            failed_counts[failed_counts >= 3].index
        )

        df["detection_reason"] = df.apply(
            get_reason, axis=1
        )

        # Overview
        total = len(df)
        normal = (df["anomaly_status"] == "NORMAL").sum()
        anomalies = (df["anomaly_status"] == "ANOMALY").sum()
        blocked = (df["security_action"] == "BLOCK").sum()

        st.success("Audit logs analyzed successfully!")

        st.header("Security Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Events", total)
        c2.metric("Normal Events", normal)
        c3.metric("Anomalies", anomalies)
        c4.metric("Blocked Events", blocked)

        st.divider()

        # Tenant filter
        st.header("Tenant-wise Monitoring")

        tenants = ["All Tenants"] + sorted(
            df["user_tenant"].dropna().unique().tolist()
        )

        selected_tenant = st.sidebar.selectbox(
            "Select Tenant",
            tenants
        )

        if selected_tenant == "All Tenants":
            view = df.copy()
        else:
            view = df[
                df["user_tenant"] == selected_tenant
            ].copy()

        st.write("Selected Tenant:", selected_tenant)
        st.metric("Events in Selected View", len(view))

        # Charts
        st.header("Anomaly Analysis")

        chart_data = pd.DataFrame({
            "Event Type": ["Normal", "Anomaly"],
            "Count": [
                normal,
                anomalies
            ]
        })

        st.bar_chart(
            chart_data,
            x="Event Type",
            y="Count"
        )

        # Reasons
        st.header("Detection Reasons")

        reason_counts = (
            df[df["anomaly_status"] == "ANOMALY"]
            ["detection_reason"]
            .value_counts()
            .reset_index()
        )

        if not reason_counts.empty:
            st.bar_chart(
                reason_counts,
                x="detection_reason",
                y="count"
            )

        # Audit logs
        st.header("Cloud Audit Logs")

        st.dataframe(
            view,
            use_container_width=True
        )

        # Alerts
        st.header("🚨 Security Alerts")

        alerts = view[
            view["security_action"] == "BLOCK"
        ]

        if not alerts.empty:
            st.error(
                str(len(alerts)) +
                " blocked events detected in this view!"
            )

            st.dataframe(
                alerts[[
                    "user_id",
                    "user_tenant",
                    "resource_tenant",
                    "detection_reason",
                    "security_action"
                ]],
                use_container_width=True
            )
        else:
            st.success("No blocked events in this view.")

        # Download
        st.download_button(
            "Download Security Report",
            df.to_csv(index=False),
            "security_report.csv",
            "text/csv"
        )

else:
    st.info("Upload your corrected security audit CSV to begin.")
