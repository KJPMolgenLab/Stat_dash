#!/opt/conda/envs/organoid_clean/bin/python3

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu, pearsonr
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import base64
import io
import plotly.io as pio


# Initialize the Dash app

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Organoid Morphology Analysis"), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select an Excel File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label='UMAP Projection', value='tab-1'),
            dcc.Tab(label='Scatter Plot', value='tab-2'),
            dcc.Tab(label='Mann-Whitney U Test', value='tab-3'),
        ]), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='tabs-content'), width=12)
    ]),
    dcc.Download(id="download-plot"),
    dcc.Download(id="download-excel")
])

# PARSING EXCEL FILE
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'xls' in filename:
        df = pd.read_excel(io.BytesIO(decoded))
    else:
        return None
    return df

# Define Callbacks for the tab switching
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'),
     Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def render_content(tab, contents, filename):
    if contents is None:
        return html.Div("Please upload an Excel file.")

    df = parse_contents(contents, filename)
    if df is None:
        return html.Div("Invalid file format.")

    exclude_columns = ['file_size', 'width', 'height']  # These are just the default metadata columns
    numeric_columns = [col for col in df.select_dtypes(include='number').columns if col not in exclude_columns]
    categorical_columns = [col for col in df.select_dtypes(include='object').columns if col not in exclude_columns][1:]

    scaler = StandardScaler()
    #df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    if tab == 'tab-1':
        color_keys = df.columns.tolist()[4:]  # Skip the first 4 columns which are metadata
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='color-key-dropdown',
                    options=[{'label': key, 'value': key} for key in color_keys if key not in ['umap_x', 'umap_y']],
                    value='circularity',
                    clearable=False
                ), width=4),
                dbc.Col(html.Button("Download Plot", id="download-umap-plot", n_clicks=0), width=2),
                dbc.Col(dcc.Download(id="download-umap-link"), width=2)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='umap-scatter-plot'), width=12)
            ])
        ])

    elif tab == 'tab-2':
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='x-axis-dropdown',
                    options=[{'label': col, 'value': col} for col in numeric_columns],
                    value=numeric_columns[-2],
                    clearable=False
                ), width=2),
                dbc.Col(dcc.Dropdown(
                    id='y-axis-dropdown',
                    options=[{'label': col, 'value': col} for col in numeric_columns],
                    value=numeric_columns[-1],
                    clearable=False
                ), width=2),
                dbc.Col(dcc.Dropdown(
                    id='color-dropdown',
                    options=[{'label': col, 'value': col} for col in df.columns],
                    value='condition',
                    clearable=False
                ), width=2),
                dbc.Col(html.Button("Download Plot", id="download-scatter-plot-btn", n_clicks=0), width=2),
                dbc.Col(dcc.Download(id="download-scatter-link"), width=2)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='scatter-plot'), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='p-value-output'), width=12)
            ])
        ])

    elif tab == 'tab-3':
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='categorical-dropdown',
                    options=[{'label': col, 'value': col} for col in categorical_columns],
                    value=categorical_columns[0],
                    clearable=False
                ), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='condition-dropdown',
                    options=[],
                    value=[],
                    clearable=True,
                    multi=True,
                    placeholder="Select conditions"
                ), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='features-dropdown',
                    options=[{'label': col, 'value': col} for col in numeric_columns],
                    value=[],
                    clearable=True,
                    multi=True,
                    placeholder="Select numerical features"
                ), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.Button("Download Excel", id="download-excel-btn", n_clicks=0), width=2)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='mann-whitney-plots'), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='mann-whitney-results'), width=12)
            ])
        ])

# Define Callbacks for the UMAP projection
@app.callback(
    Output('umap-scatter-plot', 'figure'),
    [Input('color-key-dropdown', 'value'),
     Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_umap_figure(color_key, contents, filename):
    if contents is None:
        return {}

    df = parse_contents(contents, filename)
    if df is None:
        return {}

    fig = px.scatter(
        df,
        x='umap_x',
        y='umap_y',
        color=color_key,
        hover_data={'image_name': df.index}
    )
    fig.update_layout(
        title=f"UMAP projection colored by {color_key}",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        width=800,
        height=800
    )
    return fig

# Define Callbacks for the Scatter Plot
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('p-value-output', 'children')],
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('color-dropdown', 'value'),
     Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_scatter_plot(x_axis, y_axis, color, contents, filename):
    if contents is None:
        return {}, ""

    df = parse_contents(contents, filename)
    if df is None:
        return {}, ""

    corr, p_value = pearsonr(df[x_axis], df[y_axis])
    
    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color=color,
        hover_data={x_axis: True, y_axis: True, color: True}
    )
    fig.update_layout(
        title=f"Scatter Plot of {x_axis} vs {y_axis} colored by {color}",
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        width=800,
        height=600
    )
    
    p_value_text = f"P-value between {x_axis} and {y_axis}: {p_value:.5f} with correlation coefficient {corr:.5f}"
    
    return fig, p_value_text

# Update condition dropdown based on selected categorical column
@app.callback(
    Output('condition-dropdown', 'options'),
    [Input('categorical-dropdown', 'value'),
     Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def set_condition_options(categorical_col, contents, filename):
    if contents is None or categorical_col is None:
        return []

    df = parse_contents(contents, filename)
    if df is None or categorical_col not in df.columns:
        return []

    return [{'label': val, 'value': val} for val in df[categorical_col].unique()]

# Define Callbacks for the Mann-Whitney U Test with stats tables
@app.callback(
    [Output('mann-whitney-plots', 'figure'),
     Output('mann-whitney-results', 'children')],
    [Input('condition-dropdown', 'value'),
     Input('features-dropdown', 'value'),
     Input('upload-data', 'contents'),
     Input('categorical-dropdown', 'value')],
    [State('upload-data', 'filename')]
)
def update_mann_whitney_plots(conditions, selected_features, contents, categorical_col, filename):
    if contents is None or categorical_col is None:
        return {}, "Please upload an Excel file."
    
    df = parse_contents(contents, filename)
    
    if not conditions or len(conditions) < 2:
        return {}, "Please select at least two conditions."
    if not selected_features:
        return {}, "Please select at least one feature."
    
    numeric_columns = [col for col in df.select_dtypes(include='number').columns if col not in ['file_size', 'width', 'height']]
    scaler = StandardScaler()
    #df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    results = []

    subplot_titles = [f"{feature}" for feature in selected_features]
    fig = make_subplots(rows=len(selected_features), cols=1, subplot_titles=subplot_titles)

    condition_colors = {condition: color for condition, color in zip(conditions, ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])}

    for i, feature in enumerate(selected_features):
        traces = []
        for condition in conditions:
            group = df[df[categorical_col] == condition][feature]
            traces.append(go.Violin(
                x=[condition] * len(group),
                y=group,
                name=f"{condition} - {feature}",
                box_visible=True,
                meanline_visible=True,
                line_color=condition_colors[condition],
                showlegend=False
            ))
        
        for trace in traces:
            fig.add_trace(trace, row=i+1, col=1)

        for j in range(len(conditions)):
            for k in range(j + 1, len(conditions)):
                condition1 = conditions[j]
                condition2 = conditions[k]
                group1 = df[df[categorical_col] == condition1][feature]
                group2 = df[df[categorical_col] == condition2][feature]
                stat, p_value = mannwhitneyu(group1, group2)
                results.append((feature, condition1, condition2, stat, p_value))
                
    fig.update_layout(height=300 * len(selected_features), title='Violin Plots of Numeric Features by Condition')

    desc_stats = []
    for feature in selected_features:
        for condition in conditions:
            group = df[df[categorical_col] == condition][feature]
            mean = np.mean(group)
            median = np.median(group)
            std = np.std(group)
            q1 = np.percentile(group, 25)
            q3 = np.percentile(group, 75)
            iqr = q3 - q1
            desc_stats.append((feature, condition, mean, median, std, q1, q3, iqr))

    results_table = html.Table([
        html.Thead(
            html.Tr([html.Th("Feature"), html.Th("Condition 1"), html.Th("Condition 2"), html.Th("Statistic"), html.Th("P-Value")])
        ),
        html.Tbody([
            html.Tr([
                html.Td(feature), html.Td(cond1), html.Td(cond2), html.Td(round(stat, 5)), html.Td(round(p_value, 5))
            ]) for feature, cond1, cond2, stat, p_value in results
        ])
    ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '20px'})
    
    desc_stats_table = html.Table([
        html.Thead(
            html.Tr([html.Th("Feature"), html.Th("Condition"), html.Th("Mean"), html.Th("Median"), html.Th("Std"), html.Th("Q1"), html.Th("Q3"), html.Th("IQR")])
        ),
        html.Tbody([
            html.Tr([
                html.Td(feature), html.Td(condition), html.Td(round(mean, 5)), html.Td(round(median, 5)), html.Td(round(std, 5)), html.Td(round(q1, 5)), html.Td(round(q3, 5)), html.Td(round(iqr, 5))
            ]) for feature, condition, mean, median, std, q1, q3, iqr in desc_stats
        ])
    ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '20px'})

    return fig, html.Div([results_table, html.Br(), desc_stats_table])

@app.callback(
    Output("download-umap-link", "data"),
    Input("download-umap-plot", "n_clicks"),
    [State("umap-scatter-plot", "figure")]
)
def download_umap_plot(n_clicks, umap_fig):
    if n_clicks > 0:
        img_bytes = pio.to_image(umap_fig, format='svg')
        return dcc.send_bytes(img_bytes, filename="umap_projection.svg")
    return None

@app.callback(
    Output("download-scatter-link", "data"),
    Input("download-scatter-plot-btn", "n_clicks"),
    [State("scatter-plot", "figure")]
)
def download_scatter_plot(n_clicks, scatter_fig):
    if n_clicks > 0:
        img_bytes = pio.to_image(scatter_fig, format='svg')
        return dcc.send_bytes(img_bytes, filename="scatter_plot.svg")
    return None

@app.callback(
    Output("download-excel", "data"),
    Input("download-excel-btn", "n_clicks"),
    [State('condition-dropdown', 'value'),
     State('features-dropdown', 'value'),
     Input('upload-data', 'contents'),
     State('upload-data', 'filename'),
     State('categorical-dropdown', 'value')]
)
def download_excel(n_clicks, conditions, selected_features, contents, filename, categorical_col):
    if n_clicks > 0 and contents is not None:
        df = parse_contents(contents, filename)
        if df is None:
            return None
        
        numeric_columns = [col for col in df.select_dtypes(include='number').columns if col not in ['file_size', 'width', 'height']]
        scaler = StandardScaler()
        #df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        results = []
        desc_stats = []

        for feature in selected_features:
            for condition in conditions:
                group = df[df[categorical_col] == condition][feature]
                mean = np.mean(group)
                median = np.median(group)
                std = np.std(group)
                q1 = np.percentile(group, 25)
                q3 = np.percentile(group, 75)
                iqr = q3 - q1
                desc_stats.append((feature, condition, mean, median, std, q1, q3, iqr))
            
            for j in range(len(conditions)):
                for k in range(j + 1, len(conditions)):
                    condition1 = conditions[j]
                    condition2 = conditions[k]
                    group1 = df[df[categorical_col] == condition1][feature]
                    group2 = df[df[categorical_col] == condition2][feature]
                    stat, p_value = mannwhitneyu(group1, group2)
                    results.append((feature, condition1, condition2, stat, p_value))
        
        results_df = pd.DataFrame(results, columns=["Feature", "Condition 1", "Condition 2", "Statistic", "P-Value"])
        desc_stats_df = pd.DataFrame(desc_stats, columns=["Feature", "Condition", "Mean", "Median", "Std", "Q1", "Q3", "IQR"])

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, sheet_name='Mann-Whitney U Test Results', index=False)
            desc_stats_df.to_excel(writer, sheet_name='Descriptive Statistics', index=False)
        output.seek(0)
        
        return dcc.send_bytes(output.read(), filename="mann_whitney_results.xlsx")

    return None

if __name__ == '__main__':
    app.run_server()