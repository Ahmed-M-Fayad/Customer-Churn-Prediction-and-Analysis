import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import io
import base64
import os
import requests

# Initialize the Dash app with Bootstrap for styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define the data source
csv_path = "https://raw.githubusercontent.com/Ahmed-M-Fayad/Customer-Churn-Prediction-and-Analysis/refs/heads/main/Advanced%20Data%20Analysis%20and%20Feature%20Engineering/Delivered%20Data/train_basicFeatureEng.csv"

# Define app layout
app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Customer Churn Risk Analysis Dashboard", className="text-center mb-4 mt-4"),
                html.P("Analyze factors influencing customer churn risk and identify key patterns", className="text-center mb-4"),
                html.Div(id='data-info', className="text-center mb-4")
            ], width=12)
        ]),
        
        html.Div(id='dashboard-content', children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Churn Risk Distribution", className="mt-4 mb-3"),
                    dcc.Graph(id='churn-distribution')
                ], width=6),
                dbc.Col([
                    html.H4("Churn Risk by Demographics", className="mt-4 mb-3"),
                    dcc.Dropdown(
                        id='demographic-dropdown',
                        options=[
                            {'label': 'Age', 'value': 'age'},
                            {'label': 'Gender', 'value': 'gender'},
                            {'label': 'Region', 'value': 'region_category'},
                            {'label': 'Membership', 'value': 'membership_category'}
                        ],
                        value='membership_category',
                        clearable=False
                    ),
                    dcc.Graph(id='demographic-chart')
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Customer Engagement Metrics", className="mt-4 mb-3"),
                    dcc.Dropdown(
                        id='engagement-metric-x',
                        options=[
                            {'label': 'Days Since Last Login', 'value': 'days_since_last_login'},
                            {'label': 'Points in Wallet', 'value': 'points_in_wallet'},
                            {'label': 'Avg. Transaction Value', 'value': 'avg_transaction_value'},
                            {'label': 'Points per Transaction', 'value': 'points_per_transaction'}
                        ],
                        value='days_since_last_login',
                        clearable=False
                    ),
                    dcc.Dropdown(
                        id='engagement-metric-y',
                        options=[
                            {'label': 'Points in Wallet', 'value': 'points_in_wallet'},
                            {'label': 'Days Since Last Login', 'value': 'days_since_last_login'},
                            {'label': 'Avg. Transaction Value', 'value': 'avg_transaction_value'},
                            {'label': 'Points per Transaction', 'value': 'points_per_transaction'}
                        ],
                        value='points_in_wallet',
                        clearable=False
                    ),
                    dcc.Graph(id='engagement-scatter')
                ], width=6),
                dbc.Col([
                    html.H4("Feature Importance", className="mt-4 mb-3"),
                    dcc.Graph(id='feature-importance')
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Usage Patterns", className="mt-4 mb-3"),
                    dcc.Dropdown(
                        id='usage-dropdown',
                        options=[
                            {'label': 'Medium of Operation', 'value': 'medium_of_operation'},
                            {'label': 'Internet Option', 'value': 'internet_option'},
                            {'label': 'Preferred Offer Types', 'value': 'preferred_offer_types'},
                            {'label': 'Last Visit Time of Day', 'value': 'last_visit_time_of_day'}
                        ],
                        value='medium_of_operation',
                        clearable=False
                    ),
                    dcc.Graph(id='usage-patterns')
                ], width=6),
                dbc.Col([
                    html.H4("Complaint Analysis", className="mt-4 mb-3"),
                    dcc.Graph(id='complaint-analysis')
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Time-based Analysis", className="mt-4 mb-3"),
                    dcc.Dropdown(
                        id='time-dropdown',
                        options=[
                            {'label': 'Last Visit Hour', 'value': 'last_visit_hour'},
                            {'label': 'Last Visit Time of Day', 'value': 'last_visit_time_of_day'},
                            {'label': 'Last Visit AM/PM', 'value': 'last_visit_AMPM'},
                            {'label': 'Joining Day', 'value': 'joining_day_name'},
                            {'label': 'Weekend vs Weekday', 'value': 'is_weekend'}
                        ],
                        value='last_visit_time_of_day',
                        clearable=False
                    ),
                    dcc.Graph(id='time-analysis')
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Customer Segmentation Analysis", className="mt-4 mb-3"),
                    dcc.Dropdown(
                        id='segment-dropdown-1',
                        options=[
                            {'label': 'Membership Category', 'value': 'membership_category'},
                            {'label': 'Region Category', 'value': 'region_category'},
                            {'label': 'Medium of Operation', 'value': 'medium_of_operation'},
                            {'label': 'Joined Through Referral', 'value': 'joined_through_referral'}
                        ],
                        value='membership_category',
                        clearable=False
                    ),
                    dcc.Dropdown(
                        id='segment-dropdown-2',
                        options=[
                            {'label': 'Region Category', 'value': 'region_category'},
                            {'label': 'Membership Category', 'value': 'membership_category'},
                            {'label': 'Medium of Operation', 'value': 'medium_of_operation'},
                            {'label': 'Joined Through Referral', 'value': 'joined_through_referral'}
                        ],
                        value='region_category',
                        clearable=False
                    ),
                    dcc.Graph(id='segment-heatmap')
                ], width=12)
            ])
        ])
    ])
])

# Calculate feature importance using correlation with churn risk score
def calculate_feature_importance(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'churn_risk_score' and col != 'customer_id']
    
    importance = {}
    for col in numeric_cols:
        importance[col] = abs(df[[col, 'churn_risk_score']].corr().iloc[0, 1])
    
    importance = {k: v for k, v in importance.items() if not pd.isna(v)}
    importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True)[:10])
    
    return importance

# Callback to display data information
@app.callback(
    Output('data-info', 'children'),
    Input('dashboard-content', 'children')
)
def display_data_info(content):
    try:
        return html.Div([
            html.H5(f'Data loaded from GitHub repository', style={'color': 'green'}),
            html.P(f'Shape: {df.shape[0]} rows, {df.shape[1]} columns'),
            html.P('Dataset loaded successfully!', style={'font-weight': 'bold', 'color': 'green'})
        ])
    except Exception as e:
        return html.Div([
            html.Div('Error loading data:', style={'color': 'red', 'font-weight': 'bold'}),
            html.Div(str(e), style={'color': 'red'})
        ])

# Callbacks for dashboard visualizations
@app.callback(
    Output('churn-distribution', 'figure'),
    Input('dashboard-content', 'children')
)
def update_churn_distribution(content):
    # Create churn risk score distribution
    fig = px.histogram(
        df, 
        x='churn_risk_score',
        nbins=5,
        color_discrete_sequence=['#3366CC'],
        title='Distribution of Churn Risk Scores',
        labels={'churn_risk_score': 'Churn Risk Score'}
    )
    
    # Add mean line
    mean_churn = df['churn_risk_score'].mean()
    fig.add_vline(x=mean_churn, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_churn:.2f}",
                  annotation_position="top")
    
    fig.update_layout(
        xaxis_title="Churn Risk Score",
        yaxis_title="Count",
        bargap=0.1
    )
    
    return fig

@app.callback(
    Output('demographic-chart', 'figure'),
    Input('demographic-dropdown', 'value')
)
def update_demographic_chart(selected_demographic):
    try:
        # Group by selected demographic
        demographic_groups = df.groupby(selected_demographic)['churn_risk_score'].mean().reset_index()
        demographic_counts = df.groupby(selected_demographic).size().reset_index(name='count')
        demographic_data = pd.merge(demographic_groups, demographic_counts, on=selected_demographic)
        demographic_data = demographic_data.sort_values('churn_risk_score', ascending=False)
        
        # Create a bar chart
        fig = px.bar(
            demographic_data,
            x=selected_demographic,
            y='churn_risk_score',
            color='churn_risk_score',
            height=400,
            color_continuous_scale=px.colors.sequential.Viridis,
            labels={selected_demographic: selected_demographic.replace('_', ' ').title(), 
                    'churn_risk_score': 'Avg. Churn Risk Score',
                    'count': 'Number of Customers'}
        )
        
        fig.update_layout(
            xaxis_title=selected_demographic.replace('_', ' ').title(),
            yaxis_title="Average Churn Risk Score",
            coloraxis_showscale=True,
            coloraxis_colorbar=dict(title="Churn Risk")
        )
        
        # Add text annotations for counts
        fig.update_traces(text=demographic_data['count'], textposition='outside')
        
        return fig
    except Exception as e:
        print(f"Error in demographic chart: {e}")
        # Return empty figure with error message
        return {
            'data': [],
            'layout': {
                'title': 'Error generating demographic chart',
                'annotations': [{
                    'text': f'Error: {str(e)}',
                    'showarrow': False,
                    'font': {'color': 'red'}
                }]
            }
        }

@app.callback(
    Output('engagement-scatter', 'figure'),
    [Input('engagement-metric-x', 'value'),
     Input('engagement-metric-y', 'value')]
)
def update_engagement_scatter(x_metric, y_metric):
    try:
        # First check if the columns exist
        missing_cols = []
        if x_metric not in df.columns:
            missing_cols.append(x_metric)
        if y_metric not in df.columns:
            missing_cols.append(y_metric)
            
        if missing_cols:
            return {
                'data': [],
                'layout': {
                    'title': 'Missing columns in dataset',
                    'annotations': [{
                        'text': f"Columns not found in dataset: {', '.join(missing_cols)}",
                        'showarrow': False,
                        'font': {'color': 'red'}
                    }]
                }
            }
            
        # Ensure data is not too large - sample if necessary
        sample_df = df
        if len(df) > 5000:  # If data is large, sample it
            sample_df = df.sample(5000, random_state=42)
            
        # Filter out any extreme outliers for better visualization
        q_low_x = sample_df[x_metric].quantile(0.01)
        q_high_x = sample_df[x_metric].quantile(0.99)
        q_low_y = sample_df[y_metric].quantile(0.01)
        q_high_y = sample_df[y_metric].quantile(0.99)
        
        filtered_df = sample_df[
            (sample_df[x_metric] >= q_low_x) & 
            (sample_df[x_metric] <= q_high_x) &
            (sample_df[y_metric] >= q_low_y) & 
            (sample_df[y_metric] <= q_high_y)
        ]
        
        # Create scatter plot for customer engagement metrics
        fig = px.scatter(
            filtered_df,
            x=x_metric,
            y=y_metric,
            color='churn_risk_score',
            opacity=0.7,
            hover_data=['membership_category'],
            color_continuous_scale=px.colors.sequential.Plasma,
            labels={
                x_metric: x_metric.replace('_', ' ').title(),
                y_metric: y_metric.replace('_', ' ').title(),
                'churn_risk_score': 'Churn Risk Score'
            }
        )
        
        fig.update_layout(
            xaxis_title=x_metric.replace('_', ' ').title(),
            yaxis_title=y_metric.replace('_', ' ').title(),
            coloraxis_colorbar=dict(title="Churn Risk")
        )
        
        return fig
    except Exception as e:
        print(f"Error in engagement scatter: {e}")
        # Return empty figure with error message
        return {
            'data': [],
            'layout': {
                'title': 'Error generating engagement scatter plot',
                'annotations': [{
                    'text': f'Error: {str(e)}',
                    'showarrow': False,
                    'font': {'color': 'red'}
                }]
            }
        }

@app.callback(
    Output('feature-importance', 'figure'),
    Input('dashboard-content', 'children')
)
def update_feature_importance(content):
    # Calculate feature importance
    importance = calculate_feature_importance(df)
    
    # Create horizontal bar chart for feature importance
    features = list(importance.keys())
    values = list(importance.values())
    
    fig = px.bar(
        x=values,
        y=[f.replace('_', ' ').title() for f in features],
        orientation='h',
        color=values,
        color_continuous_scale=px.colors.sequential.Greens,
        labels={'x': 'Correlation with Churn Risk Score', 'y': 'Feature'}
    )
    
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Absolute Correlation with Churn Risk",
        yaxis_title="",
        coloraxis_showscale=False
    )
    
    return fig

@app.callback(
    Output('usage-patterns', 'figure'),
    Input('usage-dropdown', 'value')
)
def update_usage_patterns(selected_usage):
    try:
        # Group by usage pattern and calculate average churn risk
        usage_groups = df.groupby(selected_usage)['churn_risk_score'].mean().reset_index()
        usage_counts = df.groupby(selected_usage).size().reset_index(name='count')
        usage_data = pd.merge(usage_groups, usage_counts, on=selected_usage)
        usage_data = usage_data.sort_values('churn_risk_score', ascending=False)
        
        # Convert count to string for text display
        usage_data['count_text'] = usage_data['count'].astype(str) + ' users'
        
        # Create horizontal bar chart - use go.Figure instead of px.bar
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=usage_data[selected_usage],
            x=usage_data['churn_risk_score'],
            orientation='h',
            text=usage_data['count_text'],
            textposition='outside',
            marker=dict(
                color=usage_data['churn_risk_score'],
                colorscale='Oranges',
                colorbar=dict(title="Churn Risk")
            )
        ))
        
        fig.update_layout(
            yaxis_title="",
            xaxis_title="Average Churn Risk Score"
        )
        
        return fig
    except Exception as e:
        print(f"Error in usage patterns: {e}")
        # Return empty figure with error message
        return {
            'data': [],
            'layout': {
                'title': 'Error generating usage patterns chart',
                'annotations': [{
                    'text': f'Error: {str(e)}',
                    'showarrow': False,
                    'font': {'color': 'red'}
                }]
            }
        }

@app.callback(
    Output('complaint-analysis', 'figure'),
    Input('dashboard-content', 'children')
)
def update_complaint_analysis(content):
    # Get complaint related data
    if 'complaint_status' in df.columns and 'feedback' in df.columns:
        # Create sunburst chart for complaint analysis
        fig = px.sunburst(
            df,
            path=['complaint_status', 'feedback'],
            values='churn_risk_score',
            color='churn_risk_score',
            color_continuous_scale=px.colors.sequential.Reds,
            title='Complaint Status & Feedback vs Churn Risk'
        )
        
        fig.update_layout(
            coloraxis_colorbar=dict(title="Avg. Churn Risk")
        )
        
        return fig
    else:
        # Return empty figure if columns are not available
        return {}

@app.callback(
    Output('time-analysis', 'figure'),
    Input('time-dropdown', 'value')
)
def update_time_analysis(selected_time):
    try:
        # For numeric time values like hour
        if selected_time == 'last_visit_hour':
            # Group by hour and calculate average churn risk
            time_data = df.groupby(selected_time)['churn_risk_score'].mean().reset_index()
            time_counts = df.groupby(selected_time).size().reset_index(name='count')
            time_data = pd.merge(time_data, time_counts, on=selected_time)
            
            # Create line chart with markers
            fig = px.line(
                time_data,
                x=selected_time,
                y='churn_risk_score',
                markers=True,
                text='count',
                color_discrete_sequence=['#7209B7'],
                labels={
                    selected_time: selected_time.replace('_', ' ').title(),
                    'churn_risk_score': 'Avg. Churn Risk Score',
                    'count': 'Number of Customers'
                }
            )
            
            # Add size markers based on count
            fig.update_traces(
                texttemplate='%{text} users', 
                textposition='top center',
                marker=dict(size=time_data['count']/time_data['count'].max()*20+5)
            )
            
        else:
            # For categorical time values - use go.Figure instead of px.bar to avoid the error
            time_data = df.groupby(selected_time)['churn_risk_score'].mean().reset_index()
            time_counts = df.groupby(selected_time).size().reset_index(name='count')
            time_data = pd.merge(time_data, time_counts, on=selected_time)
            
            # Create a custom bar chart using go.Bar
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=time_data[selected_time],
                y=time_data['churn_risk_score'],
                text=time_data['count'].astype(str) + ' users',
                textposition='outside',
                marker=dict(
                    color=time_data['churn_risk_score'],
                    colorscale='Purples',
                    colorbar=dict(title="Churn Risk")
                )
            ))
        
        fig.update_layout(
            xaxis_title=selected_time.replace('_', ' ').title(),
            yaxis_title="Average Churn Risk Score"
        )
        
        return fig
    except Exception as e:
        print(f"Error in time analysis: {e}")
        # Return empty figure with error message
        return {
            'data': [],
            'layout': {
                'title': 'Error generating time analysis chart',
                'annotations': [{
                    'text': f'Error: {str(e)}',
                    'showarrow': False,
                    'font': {'color': 'red'}
                }]
            }
        }

@app.callback(
    Output('segment-heatmap', 'figure'),
    [Input('segment-dropdown-1', 'value'),
     Input('segment-dropdown-2', 'value')]
)
def update_segment_heatmap(segment1, segment2):
    if segment1 == segment2:
        return {
            'data': [],
            'layout': {
                'annotations': [{
                    'text': 'Please select different segment variables',
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            }
        }
    
    # Create a pivot table for heatmap
    heatmap_data = df.pivot_table(
        values='churn_risk_score', 
        index=segment1, 
        columns=segment2, 
        aggfunc='mean'
    ).round(2)
    
    # Create counts heatmap for annotations
    count_data = df.pivot_table(
        values='churn_risk_score', 
        index=segment1, 
        columns=segment2, 
        aggfunc='count'
    )
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels=dict(
            x=segment2.replace('_', ' ').title(), 
            y=segment1.replace('_', ' ').title(),
            color="Avg. Churn Risk Score"
        ),
        aspect="auto"
    )
    
    # Add count annotations
    annotations = []
    for i, idx in enumerate(heatmap_data.index):
        for j, col in enumerate(heatmap_data.columns):
            annotations.append(
                dict(
                    x=col,
                    y=idx,
                    text=f"{heatmap_data.iloc[i, j]}<br>({int(count_data.iloc[i, j])} users)",
                    showarrow=False,
                    font=dict(color="black" if heatmap_data.iloc[i, j] < 3.5 else "white")
                )
            )
    
    fig.update_layout(
        annotations=annotations
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    # Load data from the GitHub URL
    csv_path = "https://raw.githubusercontent.com/Ahmed-M-Fayad/Customer-Churn-Prediction-and-Analysis/refs/heads/main/Advanced%20Data%20Analysis%20and%20Feature%20Engineering/Delivered%20Data/train_basicFeatureEng.csv"
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded data from GitHub: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
    app.run(debug=True)