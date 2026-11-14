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

        normal = (
            df["anomaly_status"] == "NORMAL"
        ).sum()

        anomalies = (
            df["anomaly_status"] == "ANOMALY"
        ).sum()

        blocked = (
            df["security_action"] == "BLOCK"
        ).sum()

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

        st.dataframe(
            df,
            use_container_width=True
        )

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
