# Customer Churn Prediction and Analysis

**Graduation Project – DEPI Initiative**  
*Ministry of Communications and Information Technology*

## Project Overview

The **Customer Churn Prediction and Analysis** project is a comprehensive machine learning (ML) initiative aimed at predicting customer churn risk to enable proactive retention strategies. Developed as a graduation project under the DEPI Initiative, it leverages a structured dataset to identify at-risk customers through data collection, exploratory data analysis (EDA), feature engineering, model development, deployment, and monitoring. The project culminated in a user-friendly **Streamlit web application** hosted on **Hugging Face Spaces**, delivering actionable insights for businesses. Comprehensive documentation for all project phases is available online at [Customer Churn Analysis and Prediction GitBook](https://ahmed-m-fayad.gitbook.io/customer-churn-analysis-and-prediction).

### Objectives

- **Business Goal**: Identify customers at risk of churning to support targeted retention campaigns.
- **Technical Goal**: Develop a robust ML model with high **F1-Score** and **ROC-AUC**, deployed as an interactive web application.
- **Success Metrics**: Achieve balanced precision and recall (**F1-Score > 0.65**) and strong discrimination ability (**ROC-AUC > 0.90**).
- **Stakeholder Impact**: Provide an intuitive tool for businesses to analyze churn risk, optimize resources, and enhance customer lifetime value.

### Key Achievements

- **Model Performance**: Deployed a tuned **Logistic Regression model** with **F1-Score of 0.6786** and **ROC-AUC of 0.9058**.
- **Deployment**: Launched a **Streamlit app** at [Hugging Face Spaces](https://huggingface.co/spaces/Mr0Diablo/customer-churn-predictor) for data upload, prediction, and visualization.
- **Documentation**: Produced detailed reports for each phase, a final presentation, and a consolidated **GitBook documentation hub**.

## Customer Churn Risk Rate Dataset

### Overview

The **Customer Churn Risk Rate** dataset comprises approximately **36,000 customer records** with **20 features**, capturing demographics, behavioral metrics, subscription details, and temporal data. Originally provided as part of the **HackerEarth Machine Learning Challenge**, it is designed to predict customer churn risk (`churn_risk_score`, multi-class levels 1–5).

### Features

- **Demographic**: `age`, `gender`, `region_category`.
- **Behavioral**: `avg_time_spent`, `avg_transaction_value`, `avg_frequency_login_days`, `days_since_last_login`, `points_in_wallet`.
- **Subscription**: `membership_category`, `preferred_offer_types`, `joined_through_referral`, `referral_id`, `feedback`.
- **Temporal**: `joining_date`, `last_visit_time`.
- **Other**: `medium_of_operation`, `internet_option`, `used_special_discount`, `offer_application_preference`, `past_complaint`.

### Target Variable

- **churn_risk_score**: Multi-class label (1–5) indicating churn risk, where **1 is lowest** and **5 is highest**.

### Dataset Sources

- **Kaggle**: [Churn Risk Rate - HackerEarth ML](https://www.kaggle.com/datasets/imsparsh/churn-risk-rate-hackerearth-ml/data)
- **HackerEarth Competition**: [Predict Customer Churn](https://www.hackerearth.com/challenges/new/competitive/hackerearth-machine-learning-challenge-predict-customer-churn/)
- **GitHub Repository**: Hosted publicly, referenced in project documentation.

### Objective

The dataset enables the development of ML models to classify customers by churn risk, leveraging historical data to identify patterns and inform retention strategies.

## Project Milestones

The project followed a structured **ML lifecycle**, with each phase producing a detailed report and contributing to the final deliverables.

1. **Data Collection & Preprocessing**

   - **Objective**: Acquire and clean the dataset for analysis.
   - **Activities**: Downloaded dataset from GitHub, performed EDA, addressed invalid values (`xxxxxxxx` in `referral_id`, `Error` in `avg_frequency_login_days`), missing values, and negative values. Dropped sensitive columns (`customer_id`, `Name`, `security_no`) for privacy.
   - **Report**: *EDA Report* – Documented data quality issues, cleaning actions, and initial insights (e.g., class imbalance, missingness).
   - **Key Insight**: Class imbalance in `churn_risk_score` required balancing techniques like **SMOTE**.

2. **Data Analysis**

   - **Objective**: Identify predictive features through statistical analysis.
   - **Activities**: Conducted **Oneway ANOVA** and **Welch’s t-Test**, visualized results with **t-SNE**, boxplots, and bar plots, confirming binary segmentation (low risk: scores 1–2; high risk: scores 3–5).
   - **Report**: *Data Analysis Report* – Detailed statistical results, visualizations, and predictors (`days_since_last_login`, `avg_time_spent`, `avg_transaction_value`).
   - **Key Insight**: Strong predictors validated for feature engineering and modeling.

3. **Feature Engineering**

   - **Objective**: Enhance model performance with new features and transformations.
   - **Activities**: Created derived metrics (`points_per_transaction`, `transaction_value_per_time_unit`), time-based features (`last_visit_hour`, `is_weekend`), and applied **one-hot**, **ordinal**, and **sentiment encoding**. Performed mathematical transformations and scaling.
   - **Report**: *Feature Engineering Summary* – Outlined new features, encoding methods, and expected impact.
   - **Key Insight**: Domain-specific features improved predictive power and model convergence.

4. **Model Development & Optimization**

   - **Objective**: Build and optimize a high-performing classification model.
   - **Activities**: Evaluated **Logistic Regression**, **Decision Tree**, **Random Forest**, **Gradient Boosting**, and **XGBoost**. Used **3-fold cross-validation**, **SMOTE**, and **RandomizedSearchCV** for tuning. Selected tuned **Logistic Regression** (**F1-Score: 0.6786**, **ROC-AUC: 0.9058**).
   - **Report**: *Model Development Report* – Covered model selection, evaluation metrics, tuning results, feature importance, and artifacts (`best_churn_model_logistic_regression.pkl`, `complete_churn_pipeline.pkl`).
   - **Key Insight**: Logistic Regression balanced performance, interpretability, and efficiency.

5. **MLOps & Deployment**

   - **Objective**: Deploy the model as an interactive web application.
   - **Activities**: Developed a **Streamlit app** hosted on **Hugging Face Spaces**, with features for CSV upload, data validation, prediction, **Plotly visualizations**, and CSV export. Tested locally and post-deployment.
   - **Report**: *Churn Predictor Process Report* – Detailed app development, testing, challenges (e.g., pie chart display, dark theme), and solutions (e.g., `@st.cache_resource`).
   - **Key Insight**: Robust error handling and visualizations enhanced user experience.

6. **Monitoring & Maintenance (Planned)**

   - **Objective**: Ensure model reliability in production.
   - **Activities**: Proposed monitoring for **F1-Score**, **ROC-AUC**, and **data drift**, with retraining triggers (**F1-Score < 0.65**) and version control (**Git**, **DVC**).
   - **Report**: Embedded in *Churn Predictor Process Report* – Outlined performance tracking, drift detection, and incident response.
   - **Key Insight**: Proactive monitoring ensures scalability and sustained performance.

7. **Documentation & Presentation**

   - **Objective**: Consolidate findings and communicate outcomes.
   - **Activities**: Produced phase reports, a final presentation, and a consolidated document report. Created a **GitBook documentation hub** at [Customer Churn Analysis and Prediction](https://ahmed-m-fayad.gitbook.io/customer-churn-analysis-and-prediction).
   - **Deliverables**:
     - *Final Presentation*: Slide deck summarizing project lifecycle, model performance, app demo, and business impact.
     - *Final Document Report*: Consolidated phase reports, technical details, and recommendations.
     - *GitBook Documentation*: Online hub with detailed phase documentation, code details, and user guide.
   - **Key Insight**: Comprehensive documentation ensures reproducibility and stakeholder alignment.

## Project Contributors

### Engineers Behind the Scene

This project was guided by the invaluable mentorship of **Eng. Ahmed Azab**, whose insights and support were instrumental throughout the journey.

It was completed through the hard work, collaboration, and dedication of the following team members:

- **Eng. Ahmed M. Fayad** – Team Leader
- **Eng. Mohammad Mostafa**
- **Eng. Ahmed Sherif**
- **Eng. Menna Elzayat**
- **Eng. Eyad Amr**

Together, we strived to deliver a meaningful and impactful contribution through teamwork and persistence.

## Getting Started

To explore or contribute to the project, follow these steps:

### Prerequisites

- **Python**: 3.8 or higher
- **Dependencies**: Listed in `requirements.txt` (e.g., **scikit-learn**, **pandas**, **numpy**, **Streamlit**, **Plotly**)
- **Environment**: **Google Colab** or local setup with **VS Code** recommended
- **Git**: For cloning the repository

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Ahmed-M-Fayad/customer-churn-prediction.git
   cd customer-churn-prediction