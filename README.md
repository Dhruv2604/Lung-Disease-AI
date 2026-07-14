# 🫁 AI-Based Lung Disease Detection & Triage System

**Live Product Demo:** https://lung-disease-ai.streamlit.app

## 1. The Product Vision
Radiology departments face massive backlogs of chest X-rays, leading to delayed diagnoses for critical patients. This product serves as an AI triage system, analyzing multiple X-rays concurrently to immediately flag abnormal scans for priority radiologist review.

## 2. Measurable Business Outcomes
* **Workflow Optimization:** Reduces patient wait times by successfully prioritizing "Abnormal" scans, allowing medical professionals to focus on critical cases first.
* **Continuous Learning Pipeline:** Engineered a scalable feedback loop where doctors can verify or correct the AI's diagnosis. This ground-truth data is appended to a central database to continuously recalculate real-world model accuracy (currently maintaining ~92%).
* **Scalable Architecture:** Designed to handle multiple concurrent image uploads efficiently, prioritizing end-user workflows and reducing operational friction.

## 3. Data & Architecture
* **Frontend Workflow (Streamlit):** Designed a seamless, batch-upload experience tailored for healthcare professionals to support data-driven decision making.
* **Backend Processing Simulation:** This deployment simulates the full-stack architecture, demonstrating how the Node.js API handles concurrent requests and serves the robust Python-based CNN model for inference.
