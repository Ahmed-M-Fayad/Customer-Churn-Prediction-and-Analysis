import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark mode appearance
st.markdown(
    """
<style>
    /* Dark mode styles */
    .main-header {
        font-size: 2.5rem;
        color: #60A5FA;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #374151;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #60A5FA;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    .info-text {
        font-size: 1.1rem;
        color: #E5E7EB;
        line-height: 1.6;
    }
    .success-box {
        padding: 1rem;
        background-color: rgba(16, 185, 129, 0.2);
        border-left: 4px solid #10B981;
        margin: 1rem 0;
        color: #D1FAE5;
    }
    .warning-box {
        padding: 1rem;
        background-color: rgba(245, 158, 11, 0.2); 
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
        color: #FEF3C7;
    }
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #2563EB;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 1.2rem;
        font-weight: 600;
        color: #E5E7EB;
    }
    div[data-testid="column"] {
        background-color: #1F2937;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #374151;
    }
    .card {
        border-radius: 0.5rem;
        background-color: #1F2937;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.3), 0 1px 2px 0 rgba(0, 0, 0, 0.2);
        padding: 1rem;
        margin: 0.5rem 0;
        color: #E5E7EB;
    }
    .metric-card {
        text-align: center;
        padding: 1.5rem 1rem;
        background: linear-gradient(135deg, #4F46E5 0%, #2563EB 100%);
        color: white;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.875rem;
        opacity: 0.9;
    }
    /* Override default Streamlit styles for dark mode */
    .css-ffhzg2 {
        background-color: #111827;
    }
    .css-1lcbmhc {
        background-color: #111827;
    }
    .css-1y4p8pa {
        max-width: 100%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    /* Fix for dark mode text in various elements */
    label, p, h1, h2, h3, h4, li {
        color: #E5E7EB !important;
    }
    .css-145kmo2 {
        color: #E5E7EB !important;
    }
    /* Dark sidebar */
    .css-1d391kg {
        background-color: #1F2937;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Title and description
st.markdown(
    '<h1 class="main-header">Customer Churn Risk Prediction</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="info-text">This interactive tool helps you predict which customers are at risk of churning based on their characteristics. Upload your customer data CSV file to get instant predictions and valuable insights.</p>',
    unsafe_allow_html=True,
)

# Required features
required_features = [
    "age",
    "gender",
    "region_category",
    "membership_category",
    "joining_date",
    "joined_through_referral",
    "preferred_offer_types",
    "medium_of_operation",
    "internet_option",
    "last_visit_time",
    "days_since_last_login",
    "avg_time_spent",
    "avg_transaction_value",
    "avg_frequency_login_days",
    "points_in_wallet",
    "used_special_discount",
    "offer_application_preference",
    "past_complaint",
    "complaint_status",
    "feedback",
]


# Function to load model
@st.cache_resource
def load_model():
    try:
        with st.spinner("Loading model..."):
            with open("complete_churn_pipeline.pkl", "rb") as file:
                model = pickle.load(file)
            st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(
            "Model file not found. Please ensure 'complete_churn_pipeline.pkl' is in the same directory."
        )
        return None


# Function to verify data
def verify_data(df):
    missing_cols = [col for col in required_features if col not in df.columns]
    if missing_cols:
        return (
            False,
            f"The following required columns are missing: {', '.join(missing_cols)}",
        )
    return True, "Data verification successful"


# Function to predict
def predict_churn(df, model):
    try:
        with st.spinner("Making predictions..."):
            # Make predictions
            predictions = model.predict(df)

            # Get probabilities if available
            try:
                probabilities = model.predict_proba(df)
                return predictions, probabilities
            except:
                return predictions, None
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None, None


# Fix for the pie chart code in the display_predictions function
def display_predictions(df, predictions, probabilities=None):
    # Add predictions to dataframe
    results_df = df.copy()
    results_df["Churn_Risk_Score"] = predictions

    # Add probabilities if available
    if probabilities is not None and probabilities.shape[1] > 1:
        for i in range(probabilities.shape[1]):
            results_df[f"Probability_Class_{i}"] = probabilities[:, i]

    # Display results
    st.markdown(
        '<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True
    )

    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Overview", "🔍 Detailed Results"])

    with tab1:
        # Summary statistics
        st.markdown(
            '<h3 style="color: #60A5FA;">Prediction Summary</h3>',
            unsafe_allow_html=True,
        )
        churn_counts = results_df["Churn_Risk_Score"].value_counts()
        churn_percentage = churn_counts / len(results_df) * 100

        # Create metrics row
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{len(results_df)}</div>
                    <div class="metric-label">Total Customers</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            if 0 in churn_counts:
                st.markdown(
                    f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #10B981 0%, #047857 100%);">
                        <div class="metric-value">{churn_counts.get(0, 0)}</div>
                        <div class="metric-label">Low Churn Risk ({churn_percentage.get(0, 0):.1f}%)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col3:
            if 1 in churn_counts:
                st.markdown(
                    f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #EF4444 0%, #B91C1C 100%);">
                        <div class="metric-value">{churn_counts.get(1, 0)}</div>
                        <div class="metric-label">High Churn Risk ({churn_percentage.get(1, 0):.1f}%)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # FIX - Create proper pie chart using Plotly with correct colors and labels
        # Ensure we have both risk categories represented in the plot
        risk_labels = {0: "Low Risk", 1: "High Risk"}

        # Create a proper data structure for the pie chart
        pie_data = []
        for risk_value in [0, 1]:  # Explicitly include both risk categories
            count = churn_counts.get(risk_value, 0)
            if risk_value in churn_counts:
                pie_data.append(
                    {
                        "risk": risk_labels[risk_value],
                        "count": count,
                        "percentage": churn_percentage.get(risk_value, 0),
                    }
                )
            else:
                # Add with zero count if category doesn't exist in predictions
                pie_data.append(
                    {"risk": risk_labels[risk_value], "count": 0, "percentage": 0.0}
                )

        # Convert to DataFrame for Plotly
        pie_df = pd.DataFrame(pie_data)

        # Set explicit colors for each category
        color_map = {
            "Low Risk": "#10B981",  # Green for low risk
            "High Risk": "#EF4444",  # Red for high risk
        }

        # Create the pie chart
        fig = px.pie(
            pie_df,
            values="count",
            names="risk",
            title="Churn Risk Distribution",
            color="risk",
            color_discrete_map=color_map,
            hole=0.4,
        )

        # Update text positioning and info
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(
                line=dict(color="#1F2937", width=2)
            ),  # Add borders to pie slices
        )

        # Update layout for better appearance
        fig.update_layout(
            legend_title="Churn Risk",
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5
            ),
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E5E7EB"),
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Detailed results table
        st.markdown(
            '<h3 style="color: #60A5FA;">Customer Predictions</h3>',
            unsafe_allow_html=True,
        )
        st.dataframe(results_df, use_container_width=True)

    return results_df


# Load the model
model = load_model()

# Sidebar for file upload and app navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/crystal-ball.png", width=100)
    st.markdown(
        "<h2 style='text-align: center; color: #60A5FA;'>Churn Predictor</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Navigation
    st.markdown("## 📂 Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    st.markdown("---")
    st.markdown("## 🧮 Model Info")
    with st.expander("About the Model"):
        st.markdown(
            """
        The churn prediction model was trained using Logistic Regression.
        
        **Model Features:**
        - Customer demographics
        - Usage patterns
        - Transaction history
        - Feedback and complaints
        """
        )

    st.markdown("---")
    st.markdown("## 💡 Tips")
    st.info(
        """
    - Make sure your CSV has all required columns
    - Data should be properly formatted
    """
    )

    st.markdown("---")
    st.markdown("### 👨‍💻 Developed By Abo Fayad")
    st.markdown("© 2025 Churn Predictor")

# Main content area
main_container = st.container()

with main_container:
    # If no file is uploaded, show data structure example by default
    if uploaded_file is None:
        st.markdown(
            '<h2 class="sub-header">Welcome to the Churn Prediction Tool</h2>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="card">
                <p class="info-text">
                This application helps you identify customers who are likely to churn, allowing you to take proactive 
                measures to retain them. By analyzing customer behavior and characteristics, the model predicts 
                which customers are at risk.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display the required features and data structure
        st.markdown(
            '<h3 class="sub-header">Required Data Structure</h3>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p>The uploaded CSV must contain the following features:</p>",
            unsafe_allow_html=True,
        )

        # Create a grid for features
        cols = st.columns(4)
        for i, feature in enumerate(required_features):
            cols[i % 4].markdown(f"✓ `{feature}`")

        # Example data structure - Expanded by default
        with st.expander("Example Data Structure", expanded=True):
            example_data = pd.DataFrame(
                {feature: [""] for feature in required_features}
            )
            st.dataframe(example_data.head(3), use_container_width=True)

        # How to use section
        st.markdown('<h3 class="sub-header">How to Use</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card">
                <ol style="padding-left: 1.5rem;">
                    <li style="margin-bottom: 0.75rem;">Upload your customer data CSV file using the sidebar</li>
                    <li style="margin-bottom: 0.75rem;">Verify that your data has all required features</li>
                    <li style="margin-bottom: 0.75rem;">Click the "Predict Churn" button to analyze your data</li>
                    <li style="margin-bottom: 0.75rem;">Explore the results and predictions</li>
                    <li>Download the predictions for further analysis</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Process the uploaded file
    else:
        # Read the CSV file
        try:
            df = pd.read_csv(uploaded_file)

            # Data analysis tabs
            tab1, tab2 = st.tabs(["📊 Data Overview", "🔮 Churn Prediction"])

            with tab1:
                # Show basic info about the uploaded data
                st.markdown(
                    '<h2 class="sub-header">Uploaded Data Overview</h2>',
                    unsafe_allow_html=True,
                )

                # Display metrics cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%);">
                            <div class="metric-value">{len(df)}</div>
                            <div class="metric-label">Total Records</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%);">
                            <div class="metric-value">{len(df.columns)}</div>
                            <div class="metric-label">Features</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col3:
                    missing_count = df.isna().sum().sum()
                    st.markdown(
                        f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);">
                            <div class="metric-value">{missing_count}</div>
                            <div class="metric-label">Missing Values</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col4:
                    duplicates = df.duplicated().sum()
                    st.markdown(
                        f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #10B981 0%, #059669 100%);">
                            <div class="metric-value">{duplicates}</div>
                            <div class="metric-label">Duplicate Rows</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Show first few rows in a nice expander
                with st.expander("Preview Data", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)

                # Data quality check
                st.markdown(
                    '<h3 class="sub-header">Data Quality Check</h3>',
                    unsafe_allow_html=True,
                )

                # Verify required columns
                missing_cols = [
                    col for col in required_features if col not in df.columns
                ]
                if missing_cols:
                    st.markdown(
                        f"""
                        <div class="warning-box">
                            <h4 style="color: #F59E0B; margin-top: 0;">Missing Columns Detected</h4>
                            <p>The following required columns are missing: {', '.join(missing_cols)}</p>
                            <p>Please ensure your data contains all required features.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="success-box">
                            <h4 style="color: #10B981; margin-top: 0;">All Required Columns Present ✓</h4>
                            <p>Your dataset contains all the necessary features for prediction.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Show data statistics
                with st.expander("Data Statistics"):
                    st.write(df.describe())

                # Data distribution visualizations for uploaded data only
                st.markdown(
                    '<h3 class="sub-header">Data Demographics</h3>',
                    unsafe_allow_html=True,
                )

                # Allow user to select a column to visualize
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

                col1, col2 = st.columns(2)

                with col1:
                    if numeric_cols:
                        selected_num_col = st.selectbox(
                            "Select a numeric feature to visualize:",
                            options=numeric_cols,
                        )

                        if selected_num_col:
                            fig = px.histogram(
                                df,
                                x=selected_num_col,
                                title=f"Distribution of {selected_num_col}",
                                color_discrete_sequence=["#4F46E5"],
                                marginal="box",
                            )
                            fig.update_layout(
                                bargap=0.1,
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#E5E7EB"),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                with col2:
                    if categorical_cols:
                        selected_cat_col = st.selectbox(
                            "Select a categorical feature to visualize:",
                            options=categorical_cols,
                        )

                        if selected_cat_col:
                            # Get value counts and limit to top 10 categories if there are many
                            value_counts = (
                                df[selected_cat_col].value_counts().reset_index()
                            )
                            value_counts.columns = [selected_cat_col, "Count"]

                            if len(value_counts) > 10:
                                value_counts = value_counts.head(10)
                                title = f"Top 10 Categories in {selected_cat_col}"
                            else:
                                title = f"Categories in {selected_cat_col}"

                            fig = px.bar(
                                value_counts,
                                x=selected_cat_col,
                                y="Count",
                                title=title,
                                color="Count",
                                color_continuous_scale="Viridis",
                            )
                            fig.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#E5E7EB"),
                            )
                            st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Verify the data
                is_valid, message = verify_data(df)

                if is_valid:
                    st.markdown(
                        """
                        <div class="success-box">
                            <h4 style="color: #10B981; margin-top: 0;">Data Validation Successful ✓</h4>
                            <p>Your data is ready for prediction. Click the button below to analyze churn risk.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Make predictions if model is loaded
                    if model:
                        predict_btn = st.button(
                            "🔮 Predict Churn Risk", key="predict_btn"
                        )

                        if predict_btn:
                            predictions, probabilities = predict_churn(df, model)

                            if predictions is not None:
                                results_df = display_predictions(
                                    df, predictions, probabilities
                                )

                                # Option to download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Predictions as CSV",
                                    data=csv,
                                    file_name="churn_predictions.csv",
                                    mime="text/csv",
                                )
                else:
                    st.markdown(
                        f"""
                        <div class="warning-box">
                            <h4 style="color: #F59E0B; margin-top: 0;">Data Validation Failed ⚠️</h4>
                            <p>{message}</p>
                            <p>Please fix these issues and upload your data again.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")
            st.markdown(
                """
                <div class="warning-box">
                    <h4 style="color: #F59E0B; margin-top: 0;">File Processing Error</h4>
                    <p>There was an error processing your file. Please make sure:</p>
                    <ul>
                        <li>The file is a valid CSV format</li>
                        <li>Column names match the required features</li>
                        <li>Data types are appropriate (numeric values for numeric fields)</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #9CA3AF; font-size: 0.8rem;">
        <p>Customer Churn Prediction Tool | Version 1.0.2 | Last Updated: May 2025</p>
    </div>
    """,
    unsafe_allow_html=True,
)
