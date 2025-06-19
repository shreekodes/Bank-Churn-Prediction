import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .churn-high {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .churn-low {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    try:
        model = joblib.load('xgb_churn_model.pkl')  # Update with your model filename
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'churn_model.pkl' is in the same directory.")
        return None

def create_feature_vector(input_data):
    """Create feature vector from input data matching the encoded dataset structure"""
    
    # Initialize all features with 0
    features = {
        'credit_score': input_data['credit_score'],
        'gender': 1 if input_data['gender'] == 'Male' else 0,
        'age': input_data['age'],
        'tenure': input_data['tenure'],
        'balance': input_data['balance'],
        'products_number': input_data['products_number'],
        'credit_card': 1 if input_data['credit_card'] else 0,
        'active_member': 1 if input_data['active_member'] else 0,
        'estimated_salary': input_data['estimated_salary'],
        'country_Germany': 1 if input_data['country'] == 'Germany' else 0,
        'country_Spain': 1 if input_data['country'] == 'Spain' else 0,
        'balance_salary_ratio': input_data['balance'] / input_data['estimated_salary'] if input_data['estimated_salary'] > 0 else 0,
        'high_balance': 1 if input_data['balance'] > 100000 else 0,  # Adjust threshold as needed
        'active_credit_combo': 1 if (input_data['active_member'] and input_data['credit_card']) else 0,
        'products_per_year': input_data['products_number'] / input_data['tenure'] if input_data['tenure'] > 0 else 0,
        'age_group_31-40': 1 if 31 <= input_data['age'] <= 40 else 0,
        'age_group_41-50': 1 if 41 <= input_data['age'] <= 50 else 0,
        'age_group_51-60': 1 if 51 <= input_data['age'] <= 60 else 0,
        'age_group_60+': 1 if input_data['age'] > 60 else 0,
        'tenure_group_Medium': 1 if 3 <= input_data['tenure'] <= 7 else 0,  # Adjust ranges as needed
        'tenure_group_High': 1 if input_data['tenure'] > 7 else 0,
    }
    
    return pd.DataFrame([features])

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Bank Churn Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("Customer Information")
    st.sidebar.markdown("---")
    
    # Input fields
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
        products_number = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0, step=1000.0)
    
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=20, value=5, step=1)
        country = st.selectbox("Country", ["France", "Germany", "Spain"])
        credit_card = st.checkbox("Has Credit Card", value=True)
        active_member = st.checkbox("Active Member", value=True)
    
    # Prediction button
    if st.sidebar.button("Predict Churn", type="primary"):
        
        # Prepare input data
        input_data = {
            'credit_score': credit_score,
            'gender': gender,
            'age': age,
            'tenure': tenure,
            'balance': balance,
            'products_number': products_number,
            'credit_card': credit_card,
            'active_member': active_member,
            'estimated_salary': estimated_salary,
            'country': country
        }
        
        # Create feature vector
        feature_vector = create_feature_vector(input_data)
        
        # Make prediction
        prediction = model.predict(feature_vector)[0]
        prediction_proba = model.predict_proba(feature_vector)[0]
        
        # Main content area
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col2:
            # Prediction result
            churn_probability = prediction_proba[1] * 100
            
            if prediction == 1:
                st.markdown(f'<div class="prediction-box churn-high">⚠️ HIGH CHURN RISK<br>Probability: {churn_probability:.1f}%</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box churn-low">✅ LOW CHURN RISK<br>Probability: {churn_probability:.1f}%</div>', 
                           unsafe_allow_html=True)
        
        # Customer profile and insights
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Customer Profile")
            
            # Create customer profile chart
            profile_data = {
                'Metric': ['Credit Score', 'Age', 'Tenure', 'Products', 'Balance ($K)', 'Salary ($K)'],
                'Value': [credit_score, age, tenure, products_number, balance/1000, estimated_salary/1000],
                'Benchmark': [650, 40, 5, 2, 80, 70]  # Example benchmarks
            }
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Customer',
                x=profile_data['Metric'],
                y=profile_data['Value'],
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                name='Average',
                x=profile_data['Metric'],
                y=profile_data['Benchmark'],
                marker_color='orange',
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Customer vs Average Profile",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Risk Factors Analysis")
            
            # Risk factors
            risk_factors = []
            
            if age > 50:
                risk_factors.append("High age group")
            if balance == 0:
                risk_factors.append("Zero balance")
            if products_number == 1:
                risk_factors.append("Single product")
            if not active_member:
                risk_factors.append("Inactive member")
            if not credit_card:
                risk_factors.append("No credit card")
            if tenure < 2:
                risk_factors.append("Low tenure")
            if credit_score < 600:
                risk_factors.append("Low credit score")
            
            if risk_factors:
                st.warning("**Risk Factors Identified:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.success("**No major risk factors identified**")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            if prediction == 1:
                recommendations = [
                    "Offer personalized retention campaign",
                    "Provide premium customer service",
                    "Consider product bundle offers",
                    "Implement targeted engagement strategies"
                ]
            else:
                recommendations = [
                    "Continue regular engagement",
                    "Offer product upgrades",
                    "Maintain service quality",
                    "Monitor for any changes"
                ]
            
            for rec in recommendations:
                st.write(f"• {rec}")
        
        # Probability gauge
        st.markdown("---")
        st.subheader("📈 Churn Probability Gauge")
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = churn_probability,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Bank Churn Prediction System
        
        This application helps predict customer churn probability using machine learning.
        
        **How to use:**
        1. Fill in the customer information in the sidebar
        2. Click "Predict Churn" to get the prediction
        3. Review the risk analysis and recommendations
        
        **Features:**
        - Real-time churn prediction
        - Risk factor analysis
        - Customer profile visualization
        - Actionable recommendations
        """)
        
        # Sample statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card"><h3>79.6%</h3><p>Retention Rate</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card"><h3>20.4%</h3><p>Churn Rate</p></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card"><h3>10,000</h3><p>Total Customers</p></div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card"><h3>21</h3><p>Features Used</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
