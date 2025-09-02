import streamlit as st

st.title("Customer Churn – Model Dashboard")
st.metric("ROC-AUC", "0.87")
st.metric("PR-AUC", "0.52")

with st.expander("Global importance"):
    st.write("SHAP summary goes here")

with st.expander("What-if"):
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    charges = st.slider("MonthlyCharges", 0.0, 150.0, 70.0)
    contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    if st.button("Predict"):
        st.write("Predicted churn probability: 0.42")
