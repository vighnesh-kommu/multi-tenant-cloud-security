Multi-Tenant Cloud Data Isolation and Leakage Prevention

1. Project Overview

This project aims to detect suspicious access activities and prevent unauthorized data access in multi-tenant cloud environments using cloud audit log analysis.

2. Problem Statement

In multi-tenant cloud environments, multiple users or organizations share cloud infrastructure. Unauthorized cross-tenant access and suspicious user activities can lead to data leakage.

Traditional access control mechanisms may not always identify unusual access patterns. Therefore, an additional monitoring and anomaly detection mechanism is proposed.

3. Objectives

- Simulate a multi-tenant cloud environment.
- Generate cloud audit logs.
- Identify unauthorized cross-tenant access.
- Detect suspicious login activities.
- Generate security alerts.
- Display security results through a dashboard.

4. Proposed Methodology

1. Create two simulated tenants.
2. Generate audit logs for user activities.
3. Analyze user and resource tenant information.
4. Detect suspicious access activities.
5. Generate ALLOW or BLOCK decisions.
6. Display results through a Streamlit dashboard.

5. Technologies Used

- Python
- Pandas
- Google Colab
- GitHub
- Streamlit

6. Current Implementation

The prototype currently uses rule-based anomaly detection to identify cross-tenant access and failed login attempts.

The system generates a security report and displays normal and anomalous activities through a web dashboard.

7. Future Enhancements

- Machine-learning-based anomaly detection.
- LSTM-based sequence analysis.
- Improved cloud audit log integration.
- Automated security alerts.
- Stronger tenant isolation enforcement.

8. Conclusion

The prototype demonstrates how audit log analysis can help identify suspicious activities in a simulated multi-tenant cloud environment and support data leakage prevention.
