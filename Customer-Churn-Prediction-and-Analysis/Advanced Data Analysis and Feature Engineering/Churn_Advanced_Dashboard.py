import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import base64
import io

# Load data
DATA_PATH = 'Advanced Data Analysis and Feature Engineering/Delivered Data/train_basicFeatureEng.csv'
df = pd.read_csv(DATA_PATH)

# Preprocess date columns
if 'joining_date' in df.columns:
    df['joining_date'] = pd.to_datetime(df['joining_date'], errors='coerce')
if 'last_visit_time' in df.columns:
    # If last_visit_time is only time, combine with a dummy date
    try:
        df['last_visit_time'] = pd.to_datetime(df['last_visit_time'], format='%H:%M:%S', errors='coerce')
    except Exception:
        pass

# Identify churn column (assuming churn_risk_score is present)
churn_col = 'churn_risk_score'

# Ensure churn_risk_score is integer (1-5)
df[churn_col] = df[churn_col].round().astype('Int64')

# Identify columns
cat_cols = [col for col in df.select_dtypes(include='object').columns if col != churn_col]
num_cols = [col for col in df.select_dtypes(include=np.number).columns if col != churn_col]

# Helper: get risk label
risk_map = {1: 'Low Risk', 2: 'Low Risk', 3: 'High Risk', 4: 'High Risk', 5: 'High Risk'}
df['risk_level'] = df[churn_col].map(lambda x: risk_map.get(int(x), 'Unknown') if pd.notnull(x) else 'Unknown')

# Helper: get membership/feedback columns
membership_col = next((c for c in df.columns if 'membership' in c.lower()), None)
feedback_col = next((c for c in df.columns if 'feedback' in c.lower()), None)
device_col = next((c for c in df.columns if 'device' in c.lower()), None)
region_col = next((c for c in df.columns if 'region' in c.lower()), None)
gender_col = next((c for c in df.columns if 'gender' in c.lower()), None)

# Helper: get tenure/time/points columns
time_spent_col = next((c for c in df.columns if 'time_spent' in c.lower()), None)
points_col = next((c for c in df.columns if 'points' in c.lower()), None)
login_freq_col = next((c for c in df.columns if 'login' in c.lower() and 'freq' in c.lower()), None)
transaction_col = next((c for c in df.columns if 'transaction' in c.lower() and 'value' in c.lower()), None)

# App initialization
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Card component
card_style = {'background': 'white', 'borderRadius': '0.5rem', 'boxShadow': '0 2px 8px #e2e8f0', 'padding': '1.5rem', 'marginBottom': '1.5rem'}

# Sidebar filters
sidebar = dbc.Card([
    html.H5('Filters', className='mb-3'),
    html.Label('Gender'),
    dcc.Dropdown(
        id='gender-filter',
        options=[{'label': g, 'value': g} for g in sorted(df[gender_col].dropna().unique())],
        multi=True,
        value=sorted(df[gender_col].dropna().unique()),  # Select all by default
        placeholder='Select gender',
    ),
    html.Label('Region', className='mt-3'),
    dcc.Dropdown(
        id='region-filter',
        options=[{'label': r, 'value': r} for r in sorted(df[region_col].dropna().unique())],
        multi=True,
        value=sorted(df[region_col].dropna().unique()),  # Select all by default
        placeholder='Select region',
    ),
    html.Label('Membership', className='mt-3'),
    dcc.Dropdown(
        id='membership-filter',
        options=[{'label': m, 'value': m} for m in sorted(df[membership_col].dropna().unique())],
        multi=True,
        value=sorted(df[membership_col].dropna().unique()),  # Select all by default
        placeholder='Select membership',
    ),
    html.Label('Joining Date Range', className='mt-3'),
    dcc.DatePickerRange(
        id='joining-date-filter',
        min_date_allowed=df['joining_date'].min() if 'joining_date' in df.columns else None,
        max_date_allowed=df['joining_date'].max() if 'joining_date' in df.columns else None,
        start_date=df['joining_date'].min() if 'joining_date' in df.columns else None,
        end_date=df['joining_date'].max() if 'joining_date' in df.columns else None,
        display_format='YYYY-MM-DD',
    ),
    html.Label('Last Visit Time Range', className='mt-3'),
    dcc.DatePickerRange(
        id='last-visit-filter',
        min_date_allowed=None,  # Not a date, so skip for now
        max_date_allowed=None,
        start_date=None,
        end_date=None,
        display_format='YYYY-MM-DD',
        disabled=True  # Placeholder for now
    ),
    html.Hr(),
    html.Label('Correlation Matrix Features', className='mt-3'),
    dcc.Dropdown(
        id='corr-feature-select',
        options=[{'label': col, 'value': col} for col in num_cols],
        value=num_cols,  # Select all by default
        multi=True,
        placeholder='Select features for correlation matrix',
    ),
    html.Br(),
    dbc.Button('Download Filtered Data', id='download-btn', color='primary', className='mb-2 w-100'),
    dcc.Download(id='download-dataframe-csv'),
], body=True, style={'minWidth': '250px', 'maxWidth': '350px', 'marginRight': '2rem'})

# Main layout
app.layout = dbc.Container([
    html.H1('Customer Churn Analysis Dashboard', className='mb-2 mt-3'),
    html.P('Interactive visualization of customer churn patterns and risk factors', className='mb-4'),
    dbc.Row([
        dbc.Col(sidebar, width=3),
        dbc.Col([
            dbc.Tabs([
                # Overview Tab
                dbc.Tab(label='Overview', children=[
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.H5('Churn Risk by Membership Type'),
                            dcc.Graph(id='membership-churn-bar')
                        ], style=card_style), md=6),
                        dbc.Col(html.Div([
                            html.H5('Customer Feedback by Risk Level'),
                            dcc.Graph(id='feedback-churn-bar')
                        ], style=card_style), md=6),
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.H5('Days Since Last Login by Churn Score'),
                            dcc.Graph(id='tenure-churn-box')
                        ], style=card_style), md=6),
                        dbc.Col(html.Div([
                            html.H5('Key Customer Metrics by Risk Level'),
                            dcc.Graph(id='transaction-churn-box'),
                            dcc.Graph(id='time-churn-box'),
                            dcc.Graph(id='points-churn-box'),
                            dcc.Graph(id='loginfreq-churn-box'),
                        ], style=card_style), md=6),
                    ]),
                ]),
                # Correlation Matrix Tab
                dbc.Tab(label='Correlation Matrix', children=[
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.H5('Dynamic Correlation Matrix'),
                            dcc.Graph(id='dynamic-corr-matrix')
                        ], style=card_style), md=12),
                    ]),
                ]),
                # Advanced Visuals Tab (placeholders for now)
                dbc.Tab(label='Advanced Visuals', children=[
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.H5('Sunburst Hierarchy'),
                            dcc.Graph(id='sunburst-hierarchy'),
                        ], style=card_style), md=6),
                    ]),
                ]),
                # Feature Insights Tab
                dbc.Tab(label='Feature Insights', children=[
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.H5('Pairwise Scatter Matrix (Top Features)'),
                            dcc.Dropdown(
                                id='scatter-feature-select',
                                options=[{'label': col, 'value': col} for col in num_cols],
                                value=sorted(df[num_cols].corrwith(df[churn_col]).abs().sort_values(ascending=False).index)[:4],
                                multi=True,
                                placeholder='Select features for scatter matrix',
                            ),
                            dcc.Graph(id='pairwise-scatter-matrix', style={'width': '100%', 'height': '700px'}),
                            html.Div('Shows pairwise relationships between top features and churn risk. If labels overlap, try selecting fewer features.', className='text-muted'),
                        ], style=card_style), md=12),
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.H5('Feature Importance (Correlation with Churn)'),
                            dcc.Graph(id='feature-importance-corr'),
                            html.Div('Feature importance is calculated as the absolute correlation between each feature and the churn risk score.', className='text-muted'),
                        ], style=card_style), md=12),
                    ]),
                ]),
                # User Segmentation Tab
                dbc.Tab(label='User Segmentation', children=[
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.H5('KMeans Clustering'),
                            html.Label('Select features for clustering:'),
                            dcc.Dropdown(
                                id='cluster-feature-select',
                                options=[{'label': col, 'value': col} for col in num_cols],
                                value=sorted(df[num_cols].corrwith(df[churn_col]).abs().sort_values(ascending=False).index)[:3],
                                multi=True,
                            ),
                            html.Label('Number of clusters:', className='mt-2'),
                            dcc.Slider(id='n-clusters', min=2, max=6, step=1, value=3, marks={i: str(i) for i in range(2, 7)}),
                            dcc.Graph(id='cluster-visualization'),
                            html.Div('Clusters are visualized using PCA for linear dimensionality reduction. PCA is more effective than t-SNE for visualizing complex cluster structures.', className='text-muted'),
                        ], style=card_style), md=12),
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.H5('Cohort Analysis: Churn by Joining Month/Tenure'),
                            dcc.RadioItems(
                                id='cohort-xaxis',
                                options=[
                                    {'label': 'Joining Month', 'value': 'join_month'}
                                ],
                                value='join_month',
                                inline=True
                            ),
                            dcc.Graph(id='cohort-analysis'),
                            html.Div('Shows churn risk distribution by cohort.', className='text-muted'),
                        ], style=card_style), md=12),
                    ]),
                ]),
            ]),
        ], width=9),
    ]),
    html.Div('Data source: train_basicFeatureEng.csv', className='mt-4 text-muted'),
], fluid=True)

# --- Callbacks ---

def filter_df(gender, region, membership, joining_start, joining_end):
    dff = df.copy()
    if gender:
        dff = dff[dff[gender_col].isin(gender)]
    if region:
        dff = dff[dff[region_col].isin(region)]
    if membership:
        dff = dff[dff[membership_col].isin(membership)]
    if joining_start and joining_end:
        dff = dff[(dff['joining_date'] >= joining_start) & (dff['joining_date'] <= joining_end)]
    return dff

def empty_fig(msg):
    return go.Figure(layout={"annotations": [{"text": msg, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 18}}]})

@app.callback(
    Output('membership-churn-bar', 'figure'),
    Output('feedback-churn-bar', 'figure'),
    Output('tenure-churn-box', 'figure'),
    Output('transaction-churn-box', 'figure'),
    Output('time-churn-box', 'figure'),
    Output('points-churn-box', 'figure'),
    Output('loginfreq-churn-box', 'figure'),
    Input('gender-filter', 'value'),
    Input('region-filter', 'value'),
    Input('membership-filter', 'value'),
    Input('joining-date-filter', 'start_date'),
    Input('joining-date-filter', 'end_date'),
)
def update_overview(gender, region, membership, joining_start, joining_end):
    dff = filter_df(gender, region, membership, joining_start, joining_end)
    if dff.empty:
        msg = "No data for selected filters."
        return (empty_fig(msg),) * 7
    fig1 = px.histogram(dff, x=membership_col, color='risk_level', barmode='group',
                        category_orders={'risk_level': ['Low Risk', 'High Risk']},
                        title='Churn Risk by Membership Type')
    fig2 = px.histogram(dff, x=feedback_col, color='risk_level', barmode='group',
                        category_orders={'risk_level': ['Low Risk', 'High Risk']},
                        title='Customer Feedback by Risk Level')
    fig3 = px.box(dff, x=churn_col, y=time_spent_col, color='risk_level',
                  title='Days Since Last Login by Churn Score')
    fig4 = px.box(dff, x='risk_level', y=transaction_col, points='all', title='Transaction Value by Risk Level')
    fig5 = px.box(dff, x='risk_level', y=time_spent_col, points='all', title='Time Spent by Risk Level')
    fig6 = px.box(dff, x='risk_level', y=points_col, points='all', title='Wallet Points by Risk Level')
    fig7 = px.box(dff, x='risk_level', y=login_freq_col, points='all', title='Login Frequency by Risk Level')
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7

@app.callback(
    Output('dynamic-corr-matrix', 'figure'),
    Input('corr-feature-select', 'value'),
    Input('gender-filter', 'value'),
    Input('region-filter', 'value'),
    Input('membership-filter', 'value'),
    Input('joining-date-filter', 'start_date'),
    Input('joining-date-filter', 'end_date'),
)
def update_corr_matrix(selected_features, gender, region, membership, joining_start, joining_end):
    dff = filter_df(gender, region, membership, joining_start, joining_end)
    if not selected_features or dff.empty:
        return empty_fig("No data or features selected.")
    corr = dff[selected_features + [churn_col]].corr()
    fig = px.imshow(corr, text_auto='.2f', aspect='auto', title='Correlation Matrix (Selected Features)')
    return fig

@app.callback(
    Output('download-dataframe-csv', 'data'),
    Input('download-btn', 'n_clicks'),
    State('gender-filter', 'value'),
    State('region-filter', 'value'),
    State('membership-filter', 'value'),
    State('joining-date-filter', 'start_date'),
    State('joining-date-filter', 'end_date'),
    prevent_initial_call=True
)
def download_filtered(n_clicks, gender, region, membership, joining_start, joining_end):
    dff = filter_df(gender, region, membership, joining_start, joining_end)
    return dcc.send_data_frame(dff.to_csv, 'filtered_churn_data.csv')

@app.callback(
    Output('feature-importance-corr', 'figure'),
    Input('gender-filter', 'value'),
    Input('region-filter', 'value'),
    Input('membership-filter', 'value'),
    Input('joining-date-filter', 'start_date'),
    Input('joining-date-filter', 'end_date'),
)
def update_feature_importance_corr(gender, region, membership, joining_start, joining_end):
    dff = filter_df(gender, region, membership, joining_start, joining_end)
    if dff.empty:
        return empty_fig('No data for selected filters.')
    corrs = dff[num_cols].corrwith(dff[churn_col]).abs().sort_values(ascending=False)
    fig = px.bar(x=corrs.index, y=corrs.values, labels={'x': 'Feature', 'y': '|Correlation with Churn|'},
                 title='Feature Importance (Correlation with Churn)')
    return fig

@app.callback(
    Output('sunburst-hierarchy', 'figure'),
    Input('gender-filter', 'value'),
    Input('region-filter', 'value'),
    Input('membership-filter', 'value'),
    Input('joining-date-filter', 'start_date'),
    Input('joining-date-filter', 'end_date'),
)
def update_sunburst(gender, region, membership, joining_start, joining_end):
    dff = filter_df(gender, region, membership, joining_start, joining_end)
    path = [region_col, membership_col, 'risk_level']
    fig = px.sunburst(dff, path=path, values=None, color='risk_level', title='Churn Hierarchy by Region, Membership, Risk')
    return fig

@app.callback(
    Output('pairwise-scatter-matrix', 'figure'),
    Input('scatter-feature-select', 'value'),
    Input('gender-filter', 'value'),
    Input('region-filter', 'value'),
    Input('membership-filter', 'value'),
    Input('joining-date-filter', 'start_date'),
    Input('joining-date-filter', 'end_date'),
)
def update_scatter_matrix(features, gender, region, membership, joining_start, joining_end):
    dff = filter_df(gender, region, membership, joining_start, joining_end)
    if not features or len(features) < 2 or dff.empty:
        return empty_fig('Select at least two features and ensure data is available.')
    fig = px.scatter_matrix(
        dff,
        dimensions=features,
        color='risk_level',
        title='Pairwise Scatter Matrix',
        labels={col: col for col in features},
        height=700,
        width=1200
    )
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig

@app.callback(
    Output('cluster-visualization', 'figure'),
    Input('cluster-feature-select', 'value'),
    Input('n-clusters', 'value'),
    Input('gender-filter', 'value'),
    Input('region-filter', 'value'),
    Input('membership-filter', 'value'),
    Input('joining-date-filter', 'start_date'),
    Input('joining-date-filter', 'end_date'),
)
def update_cluster_plot(features, n_clusters, gender, region, membership, joining_start, joining_end):
    dff = filter_df(gender, region, membership, joining_start, joining_end)
    if not features or len(features) < 2 or dff.empty:
        return empty_fig('Select at least two features and ensure data is available.')
    X = dff[features].dropna()
    if X.shape[0] < n_clusters:
        return empty_fig('Not enough data for the selected number of clusters.')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    dff = dff.loc[X.index].copy()
    dff['cluster'] = clusters.astype(str)
    # Use PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    dff['pca1'] = X_pca[:, 0]
    dff['pca2'] = X_pca[:, 1]
    fig = px.scatter(dff, x='pca1', y='pca2', color='cluster', symbol='risk_level',
                     title='KMeans Clusters (PCA 2D)',
                     hover_data=features + ['risk_level'])
    return fig

@app.callback(
    Output('cohort-analysis', 'figure'),
    Input('cohort-xaxis', 'value'),
    Input('gender-filter', 'value'),
    Input('region-filter', 'value'),
    Input('membership-filter', 'value'),
    Input('joining-date-filter', 'start_date'),
    Input('joining-date-filter', 'end_date'),
)
def update_cohort_plot(xaxis, gender, region, membership, joining_start, joining_end):
    dff = filter_df(gender, region, membership, joining_start, joining_end)
    if dff.empty:
        return empty_fig('No data for selected filters.')
    if xaxis == 'join_month' and 'joining_date' in dff.columns:
        dff['join_month'] = dff['joining_date'].dt.to_period('M').astype(str)
        group = dff.groupby(['join_month', 'risk_level']).size().reset_index(name='count')
        fig = px.bar(group, x='join_month', y='count', color='risk_level', barmode='group',
                     title='Churn by Joining Month')
        return fig
    else:
        return empty_fig('No data or invalid selection.')

if __name__ == '__main__':
    app.run(debug=True) 