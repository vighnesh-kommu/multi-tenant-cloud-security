
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
