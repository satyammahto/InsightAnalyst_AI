import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Enterprise Light Theme Config
THEME_TEMPLATE = 'plotly_white'
COLOR_PRIMARY = '#0284C7'    # Sky Blue
COLOR_SUCCESS = '#059669'    # Emerald
COLOR_PURPLE = '#7C3AED'     # Violet
COLOR_DANGER = '#E11D48'     # Rose
COLOR_AMBER = '#D97706'      # Amber

CLUSTER_PALETTE = ['#059669', '#0284C7', '#7C3AED', '#EA580C', '#EC4899', '#2563EB']

COMMON_COLUMN_MAP = {
    'ORDERNUMBER': 'Order Number',
    'QUANTITYORDERED': 'Quantity Ordered',
    'PRICEEACH': 'Price',
    'ORDERLINENUMBER': 'Order Line Number',
    'SALES': 'Sales',
    'ORDERDATE': 'Order Date',
    'STATUS': 'Status',
    'QTR_ID': 'Quarter',
    'MONTH_ID': 'Month',
    'YEAR_ID': 'Year',
    'PRODUCTLINE': 'Product Line',
    'MSRP': 'MSRP',
    'PRODUCTCODE': 'Product Code',
    'CUSTOMERNAME': 'Customer Name',
    'PHONE': 'Phone',
    'ADDRESSLINE1': 'Address Line 1',
    'ADDRESSLINE2': 'Address Line 2',
    'CITY': 'City',
    'STATE': 'State',
    'POSTALCODE': 'Postal Code',
    'COUNTRY': 'Country',
    'TERRITORY': 'Territory',
    'CONTACTLASTNAME': 'Contact Last Name',
    'CONTACTFIRSTNAME': 'Contact First Name',
    'DEALSIZE': 'Deal Size'
}

def _clean_name(col_name: str) -> str:
    """Helper to convert technical column names to clean, readable words."""
    if not col_name:
        return ""
    upper_c = str(col_name).strip().upper()
    if upper_c in COMMON_COLUMN_MAP:
        return COMMON_COLUMN_MAP[upper_c]
    words = str(col_name).replace('_', ' ').split()
    cleaned_words = []
    for w in words:
        if w.upper() in ['ID', 'MSRP', 'URL', 'USD', 'EUR', 'APAC', 'EMEA', 'NA', 'SKU', 'QTR', 'ZIP']:
            cleaned_words.append(w.upper())
        else:
            cleaned_words.append(w.capitalize())
    return " ".join(cleaned_words)

def plot_numeric_distribution(df: pd.DataFrame, col: str, simple_mode: bool = True) -> go.Figure:
    """Generates a distribution histogram with marginal boxplot."""
    clean_col = _clean_name(col)
    series = df[col].dropna()
    min_val = series.min() if not series.empty else 0
    max_val = series.max() if not series.empty else 0
    mean_val = series.mean() if not series.empty else 0

    if simple_mode:
        title = f"<b>{clean_col} Distribution</b><br><span style='font-size:12px; font-weight:normal; color:#64748B;'>Values range from {min_val:,.1f} to {max_val:,.1f} · Average: {mean_val:,.1f}</span>"
        x_title = clean_col
        y_title = "Number of Records"
        mean_label = f"Average: {mean_val:,.1f}"
    else:
        title = f"NUMERIC PARAMETRIC ANALYSIS<br><b>{col} · Distribution Histogram</b>"
        x_title = col
        y_title = "Density / Count"
        mean_label = f"Mean: {mean_val:.2f}"

    fig = px.histogram(
        df,
        x=col,
        marginal='box',
        title=title,
        color_discrete_sequence=[COLOR_PRIMARY],
        template=THEME_TEMPLATE,
        nbins=30
    )
    
    # Add average line
    fig.add_vline(x=mean_val, line_dash="dash", line_color=COLOR_SUCCESS, annotation_text=mean_label, annotation_position="top right")

    fig.update_layout(
        margin=dict(l=40, r=40, t=65, b=40),
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=x_title),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=y_title),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig

def plot_categorical_counts(df: pd.DataFrame, col: str, top_n: int = 7, simple_mode: bool = True) -> go.Figure:
    """Generates a Top-N Volume & Share horizontal bar chart."""
    clean_col = _clean_name(col)
    val_counts = df[col].value_counts().head(top_n).reset_index()
    val_counts.columns = [col, 'Count']
    total = len(df)
    val_counts['Share %'] = (val_counts['Count'] / total * 100).map('{:.1f}%'.format)
    
    if simple_mode:
        title = f"<b>Top {min(top_n, len(val_counts))} Most Common {clean_col}</b><br><span style='font-size:12px; font-weight:normal; color:#64748B;'>Most frequent categories and their share of all records</span>"
        x_title = "Number of Records"
        y_title = clean_col
    else:
        title = f"CATEGORICAL CARDINALITY ANALYSIS<br><b>{col} · Top-{min(top_n, len(val_counts))} Volume & Share</b>"
        x_title = "Count"
        y_title = col

    fig = px.bar(
        val_counts,
        x='Count',
        y=col,
        text='Share %',
        orientation='h',
        title=title,
        color='Count',
        color_continuous_scale='Blues',
        template=THEME_TEMPLATE
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        margin=dict(l=40, r=60, t=65, b=40),
        yaxis=dict(autorange="reversed", showgrid=False, title=y_title),
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=x_title),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        coloraxis_showscale=False,
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig

def plot_date_trend(df: pd.DataFrame, date_col: str, numeric_col: str = None, simple_mode: bool = True) -> go.Figure:
    """Generates an activity trend chart over time."""
    clean_date = _clean_name(date_col)
    clean_num = _clean_name(numeric_col) if numeric_col else "Records"

    temp_df = df.dropna(subset=[date_col]).copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
    
    date_range_days = (temp_df[date_col].max() - temp_df[date_col].min()).days
    freq = 'D' if date_range_days <= 60 else ('ME' if date_range_days <= 730 else 'YE')
    
    try:
        if numeric_col and numeric_col in temp_df.columns:
            trend = temp_df.set_index(date_col).resample(freq)[numeric_col].mean().reset_index()
            y_val = numeric_col
            if simple_mode:
                title = f"<b>{clean_num} Over Time</b><br><span style='font-size:12px; font-weight:normal; color:#64748B;'>Average {clean_num} across dates ({clean_date})</span>"
                y_label = f"Average {clean_num}"
            else:
                title = f"TEMPORAL DECOMPOSITION<br><b>{numeric_col} Volume & Rolling Trend ({date_col})</b>"
                y_label = numeric_col
        else:
            trend = temp_df.set_index(date_col).resample(freq).size().reset_index(name='Volume')
            y_val = 'Volume'
            if simple_mode:
                title = f"<b>Activity Over Time</b><br><span style='font-size:12px; font-weight:normal; color:#64748B;'>Record volume across dates ({clean_date})</span>"
                y_label = "Number of Records"
            else:
                title = f"TEMPORAL DECOMPOSITION<br><b>Transaction Volume Trend ({date_col})</b>"
                y_label = 'Volume'
    except Exception:
        temp_df['Date_Group'] = temp_df[date_col].dt.date
        if numeric_col and numeric_col in temp_df.columns:
            trend = temp_df.groupby('Date_Group')[numeric_col].mean().reset_index()
            trend.columns = [date_col, numeric_col]
            y_val = numeric_col
            if simple_mode:
                title = f"<b>{clean_num} Daily Trend</b><br><span style='font-size:12px; font-weight:normal; color:#64748B;'>Daily average of {clean_num}</span>"
                y_label = f"Average {clean_num}"
            else:
                title = f"TEMPORAL DECOMPOSITION<br><b>{numeric_col} Daily Average ({date_col})</b>"
                y_label = numeric_col
        else:
            trend = temp_df.groupby('Date_Group').size().reset_index(name='Volume')
            trend.columns = [date_col, 'Volume']
            y_val = 'Volume'
            if simple_mode:
                title = f"<b>Daily Record Volume</b><br><span style='font-size:12px; font-weight:normal; color:#64748B;'>Daily record count</span>"
                y_label = "Number of Records"
            else:
                title = f"TEMPORAL DECOMPOSITION<br><b>Daily Transaction Volume ({date_col})</b>"
                y_label = 'Volume'

    fig = px.area(
        trend,
        x=date_col,
        y=trend.columns[1],
        title=title,
        color_discrete_sequence=['#0284C7'],
        template=THEME_TEMPLATE
    )
    fig.update_traces(line=dict(width=3), fillcolor='rgba(2, 132, 199, 0.1)')
    fig.update_layout(
        margin=dict(l=40, r=40, t=65, b=40),
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title="Timeline"),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=y_label),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig

def plot_correlation_heatmap(corr_df: pd.DataFrame, simple_mode: bool = True) -> go.Figure:
    """Generates an annotated correlation matrix heatmap."""
    z_vals = corr_df.values
    x_labels = list(corr_df.columns)
    y_labels = list(corr_df.index)
    
    annotations = []
    for i, row in enumerate(y_labels):
        for j, col in enumerate(x_labels):
            val = z_vals[i][j]
            val_str = f"{val:+.2f}" if val != 1.0 else "1.00"
            
            # Color logic: Blue for positive, pink for negative
            font_col = "#0369A1" if val > 0.3 else ("#BE123C" if val < -0.2 else "#475569")
            
            annotations.append(dict(
                x=col, y=row,
                text=f"<b>{val_str}</b>",
                font=dict(color=font_col, size=11, family="JetBrains Mono, monospace"),
                showarrow=False
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z_vals, x=x_labels, y=y_labels,
        colorscale=[[0, '#FFE4E6'], [0.5, '#F8FAFC'], [1, '#E0F2FE']],
        zmin=-1, zmax=1,
        showscale=False
    ))
    
    if simple_mode:
        title = "<b>Which Numbers Move Together?</b><br><span style='font-size:12px; font-weight:normal; color:#64748B;'>Stronger blue = move together; pink = move oppositely.</span>"
    else:
        title = "MATRIX ANALYSIS<br><b>Pearson Correlation Matrix</b>"

    fig.update_layout(
        annotations=annotations,
        title=title,
        template=THEME_TEMPLATE,
        margin=dict(l=60, r=40, t=65, b=60),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig

def plot_cluster_scatter(pca_df: pd.DataFrame, n_clusters: int = 4, simple_mode: bool = True) -> go.Figure:
    """Generates a 2D Group Scatter Plot."""
    display_df = pca_df.copy()
    if simple_mode:
        display_df['Group'] = display_df['Persona']
        color_col = 'Group'
        title = f"<b>{n_clusters} Natural Groups in Your Data</b><br><span style='font-size:12px; font-weight:normal; color:#64748B;'>AI grouped similar records based on their characteristics.</span>"
        x_label = "Pattern Dimension 1"
        y_label = "Pattern Dimension 2"
        center_label = "Group Centers (o)"
    else:
        display_df['Group'] = display_df['Cluster']
        color_col = 'Group'
        title = "UNSUPERVISED SEGMENTS<br><b>KMeans Behavioral Clustering - 2D PCA Space</b>"
        x_label = "PCA-1 Vector"
        y_label = "PCA-2 Vector"
        center_label = "Centroid Nodes (o)"

    hover_cols = [c for c in display_df.columns if c not in ['PCA1', 'PCA2', 'Cluster', 'Persona', 'Group']]
    
    fig = px.scatter(
        display_df,
        x='PCA1',
        y='PCA2',
        color=color_col,
        hover_data=['Persona'] + hover_cols[:3],
        title=title,
        color_discrete_sequence=CLUSTER_PALETTE,
        template=THEME_TEMPLATE
    )
    
    # Calculate Centroids
    centroids = display_df.groupby(color_col)[['PCA1', 'PCA2']].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=centroids['PCA1'],
        y=centroids['PCA2'],
        mode='markers',
        marker=dict(size=14, color='white', line=dict(width=3, color='#0F172A'), symbol='circle'),
        name=center_label,
        hoverinfo='skip'
    ))

    fig.update_traces(marker=dict(size=9, opacity=0.85))
    fig.update_layout(
        margin=dict(l=40, r=40, t=65, b=40),
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=x_label),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=y_label),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig

def plot_outlier_scatter(outlier_df: pd.DataFrame, x_col: str, y_col: str, simple_mode: bool = True) -> go.Figure:
    """Generates an Outlier highlight scatter plot."""
    clean_x = _clean_name(x_col)
    clean_y = _clean_name(y_col)
    
    display_df = outlier_df.copy()
    if simple_mode:
        display_df['Record Status'] = ['🔴 Unusual Record' if a == 'Isolated Anomaly' else '🟢 Normal Record' for a in display_df['Is_Outlier']]
        color_col = 'Record Status'
        color_map = {'🟢 Normal Record': '#0284C7', '🔴 Unusual Record': '#E11D48'}
        title = f"<b>Unusual Records in Your Data ({clean_x} vs {clean_y})</b><br><span style='font-size:12px; font-weight:normal; color:#64748B;'>Red dots represent records that look very different from the normal pattern.</span>"
    else:
        color_col = 'Is_Outlier'
        color_map = {'Inlier': '#0284C7', 'Isolated Anomaly': '#E11D48'}
        title = f"UNSUPERVISED OUTLIER TELEMETRY<br><b>Isolation Forest Anomaly Detection ({x_col} vs {y_col})</b>"

    fig = px.scatter(
        display_df,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_map=color_map,
        title=title,
        template=THEME_TEMPLATE
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85))
    fig.update_layout(
        margin=dict(l=40, r=40, t=65, b=40),
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=clean_x if simple_mode else x_col),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=clean_y if simple_mode else y_col),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig

def plot_custom_user_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str = None, color_col: str = None, agg_func: str = "Mean", z_col: str = None, simple_mode: bool = True) -> go.Figure:
    """
    Builds custom user-requested visualizations dynamically.
    Supports: Scatter, Bar (Aggregated), Line, Box, Violin, Histogram, Pie/Donut, 3D Scatter.
    """
    color_param = color_col if color_col and color_col != "None" else None
    y_param = y_col if y_col and y_col != "None" else None
    z_param = z_col if z_col and z_col != "None" else None

    clean_x = _clean_name(x_col) if simple_mode else x_col
    clean_y = _clean_name(y_param) if (simple_mode and y_param) else y_param
    clean_color = _clean_name(color_param) if (simple_mode and color_param) else color_param

    if simple_mode:
        title_str = f"<b>{chart_type}: {clean_x}" + (f" vs {clean_y}" if clean_y else "") + (f" (Grouped by {clean_color})" if clean_color else "") + "</b>"
    else:
        title_str = f"CUSTOM EXPLORER<br><b>{chart_type}: {x_col}" + (f" vs {y_param}" if y_param else "") + (f" (Grouped by {color_param})" if color_param else "") + "</b>"

    if chart_type == "Scatter Plot":
        fig = px.scatter(df, x=x_col, y=y_param, color=color_param, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)
        fig.update_traces(marker=dict(size=8, opacity=0.85))

    elif chart_type == "Bar Chart (Aggregated)":
        if y_param:
            if agg_func == "Sum":
                agg_df = df.groupby(x_col)[y_param].sum().reset_index()
            elif agg_func == "Count":
                agg_df = df.groupby(x_col)[y_param].count().reset_index()
            elif agg_func == "Median":
                agg_df = df.groupby(x_col)[y_param].median().reset_index()
            else:
                agg_df = df.groupby(x_col)[y_param].mean().reset_index()
            agg_label = "Average" if agg_func == "Mean" else agg_func
            title_display = f"{title_str} ({agg_label})"
            fig = px.bar(agg_df, x=x_col, y=y_param, color=color_param, title=title_display, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)
        else:
            val_counts = df[x_col].value_counts().reset_index()
            val_counts.columns = [x_col, 'Count']
            fig = px.bar(val_counts, x=x_col, y='Count', color=color_param, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)

    elif chart_type == "Line Chart":
        fig = px.line(df, x=x_col, y=y_param, color=color_param, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)

    elif chart_type == "Box Plot":
        fig = px.box(df, x=x_col, y=y_param, color=color_param, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)

    elif chart_type == "Violin Plot":
        fig = px.violin(df, x=x_col, y=y_param, color=color_param, box=True, points="all", title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)

    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_col, y=y_param, color=color_param, marginal="rug", title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)

    elif chart_type == "Pie / Donut Chart":
        if y_param:
            fig = px.pie(df, names=x_col, values=y_param, color=color_param, hole=0.4, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)
        else:
            fig = px.pie(df, names=x_col, hole=0.4, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)

    elif chart_type == "3D Scatter Plot" and z_param:
        fig = px.scatter_3d(df, x=x_col, y=y_param, z=z_param, color=color_param, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)
    else:
        fig = px.scatter(df, x=x_col, y=y_param, color=color_param, title=title_str, template=THEME_TEMPLATE, color_discrete_sequence=CLUSTER_PALETTE)

    fig.update_layout(
        margin=dict(l=40, r=40, t=65, b=40),
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=clean_x),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=clean_y if clean_y else "Value"),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig


def plot_fuzzy_severity_distribution(severity_counts: dict) -> go.Figure:
    """Generates a clean donut chart for Fuzzy Anomaly Severity Risk Overview."""
    labels = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
    values = [
        severity_counts.get('Low', 0),
        severity_counts.get('Medium', 0),
        severity_counts.get('High', 0),
        severity_counts.get('Critical', 0)
    ]
    colors = ['#059669', '#D97706', '#EA580C', '#E11D48']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])

    fig.update_layout(
        title="<b>Anomaly Risk Severity Distribution</b><br><span style='font-size:12px; color:#64748B;'>Fuzzy inference classification of unusual records</span>",
        template=THEME_TEMPLATE,
        margin=dict(l=20, r=20, t=60, b=20),
        height=320,
        showlegend=True,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig


def plot_fuzzy_scatter(outlier_df: pd.DataFrame) -> go.Figure:
    """Plots Anomaly Intensity vs. Fuzzy Severity Score, colored by Severity Label."""
    anom_df = outlier_df[outlier_df.get('Severity_Label', 'Normal') != 'Normal'].copy()
    if anom_df.empty:
        anom_df = outlier_df.head(20).copy()

    color_map = {
        'Low': '#059669',
        'Medium': '#D97706',
        'High': '#EA580C',
        'Critical': '#E11D48'
    }

    fig = px.scatter(
        anom_df,
        x='Anomaly_Intensity',
        y='Severity_Score',
        color='Severity_Label',
        color_discrete_map=color_map,
        hover_data=['Severity_Explanation', 'Feature_Deviation', 'Business_Impact'],
        title="<b>Anomaly Intensity vs. Fuzzy Severity Score</b><br><span style='font-size:12px; color:#64748B;'>Demonstrating gradual fuzzy inference transitions</span>",
        template=THEME_TEMPLATE
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color='#FFFFFF')))
    fig.update_layout(
        xaxis=dict(title="Anomaly Intensity (0–100)", showgrid=True, gridcolor='#F1F5F9'),
        yaxis=dict(title="Fuzzy Severity Score (0–100)", showgrid=True, gridcolor='#F1F5F9'),
        margin=dict(l=40, r=40, t=65, b=40),
        height=350,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig


def plot_fuzzy_membership_functions(curves_data: dict) -> go.Figure:
    """Plots overlapping input and output fuzzy membership curves for viva/demo presentation."""
    out_curves = curves_data.get('output_curves', {})
    u = out_curves.get('universe', np.linspace(0, 100, 201))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u, y=out_curves['low'], mode='lines', name='Low [0–35]', line=dict(color='#059669', width=2.5)))
    fig.add_trace(go.Scatter(x=u, y=out_curves['medium'], mode='lines', name='Medium [20–62]', line=dict(color='#D97706', width=2.5)))
    fig.add_trace(go.Scatter(x=u, y=out_curves['high'], mode='lines', name='High [48–85]', line=dict(color='#EA580C', width=2.5)))
    fig.add_trace(go.Scatter(x=u, y=out_curves['critical'], mode='lines', name='Critical [72–100]', line=dict(color='#E11D48', width=2.5)))

    fig.update_layout(
        title="<b>Mamdani Fuzzy Membership Functions (Severity Output)</b><br><span style='font-size:12px; color:#64748B;'>Overlapping fuzzy sets demonstrating gradual membership transitions</span>",
        template=THEME_TEMPLATE,
        xaxis=dict(title="Severity Universe [0 to 100]", showgrid=True, gridcolor='#F1F5F9'),
        yaxis=dict(title="Membership Grade (μ)", range=[0, 1.05], showgrid=True, gridcolor='#F1F5F9'),
        margin=dict(l=40, r=40, t=65, b=40),
        height=280,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=11, color="#334155")
    )
    return fig


def plot_elbow_silhouette(k_diagnostics: dict) -> go.Figure:
    """Plots dual-axis Elbow (Inertia) & Silhouette Optimization curve."""
    k_vals = k_diagnostics.get('k_values', [2, 3, 4, 5, 6, 7, 8])
    inertias = k_diagnostics.get('inertias', [])
    silhouettes = k_diagnostics.get('silhouettes', [])
    rec_k = k_diagnostics.get('recommended_k', 4)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Left Y-Axis: Inertia (Elbow)
    fig.add_trace(
        go.Scatter(
            x=k_vals,
            y=inertias,
            name="Inertia (Elbow)",
            mode='lines+markers',
            marker=dict(size=8, color="#0284C7"),
            line=dict(color="#0284C7", width=2.5)
        ),
        secondary_y=False
    )

    # Right Y-Axis: Silhouette Score
    fig.add_trace(
        go.Scatter(
            x=k_vals,
            y=silhouettes,
            name="Silhouette Score",
            mode='lines+markers',
            marker=dict(size=8, color="#059669", symbol="diamond"),
            line=dict(color="#059669", width=2.5, dash="dash")
        ),
        secondary_y=True
    )

    # Add vertical line for recommended K
    fig.add_vline(
        x=rec_k,
        line_dash="dot",
        line_color="#7C3AED",
        annotation_text=f"AI Recommendation: k={rec_k}",
        annotation_position="top left",
        annotation_font=dict(color="#7C3AED", size=11, family="Inter, sans-serif")
    )

    fig.update_layout(
        title=f"<b>Optimal Cluster Diagnostics (k=2 to {max(k_vals)})</b><br><span style='font-size:12px; color:#64748B;'>Left: Inertia (lower is tighter) · Right: Silhouette Score (higher indicates cleaner cluster boundaries)</span>",
        template=THEME_TEMPLATE,
        margin=dict(l=40, r=40, t=65, b=40),
        height=320,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        xaxis=dict(title="Number of Clusters (k)", tickmode='linear', tick0=2, dtick=1, showgrid=True, gridcolor='#F1F5F9'),
        font=dict(family="Inter, sans-serif", size=11, color="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Inertia (Sum of Squares)", secondary_y=False, showgrid=True, gridcolor='#F1F5F9')
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True, showgrid=False)
    return fig


def plot_cluster_radar(radar_profiles: dict, simple_mode: bool = True) -> go.Figure:
    """Plots multi-axis Polar Radar chart comparing cluster feature profiles."""
    if not radar_profiles:
        return go.Figure()

    fig = go.Figure()
    palette = ['#059669', '#0284C7', '#7C3AED', '#EA580C', '#EC4899', '#2563EB', '#14B8A6', '#F59E0B']

    for i, (cluster_name, feat_dict) in enumerate(radar_profiles.items()):
        # Limit to top 8 features for clean readability if many features exist
        items = list(feat_dict.items())[:8]
        categories = [k for k, _ in items]
        values = [v for _, v in items]
        # Close the loop
        categories.append(categories[0])
        values.append(values[0])

        col_color = palette[i % len(palette)]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=cluster_name.split(':')[0],
            line=dict(color=col_color, width=2),
            opacity=0.65
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                tickfont=dict(size=9, color="#64748B"),
                gridcolor="#E2E8F0"
            ),
            angularaxis=dict(
                tickfont=dict(size=10.5, family="Inter, sans-serif", color="#1E293B"),
                gridcolor="#E2E8F0"
            ),
            bgcolor="#FAFAFA"
        ),
        title="<b>Multi-Feature Behavioral Fingerprint Across Clusters</b><br><span style='font-size:12px; color:#64748B;'>Normalized 0–100 scale comparing cluster dimensions</span>",
        template=THEME_TEMPLATE,
        margin=dict(l=40, r=40, t=65, b=40),
        height=380,
        paper_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=11, color="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig


def plot_outliers_by_cluster(outliers_by_cluster_df: pd.DataFrame, simple_mode: bool = True) -> go.Figure:
    """Plots stacked bar chart showing outlier risk severity breakdown across clusters."""
    if outliers_by_cluster_df is None or outliers_by_cluster_df.empty:
        return go.Figure()

    fig = go.Figure()
    clusters = list(outliers_by_cluster_df['Cluster'])

    severity_layers = [
        ('Low', '#059669', 'Low Risk'),
        ('Medium', '#D97706', 'Medium Risk'),
        ('High', '#EA580C', 'High Risk'),
        ('Critical', '#E11D48', 'Critical Risk')
    ]

    for col_key, color_code, label_name in severity_layers:
        if col_key in outliers_by_cluster_df.columns:
            fig.add_trace(go.Bar(
                name=label_name,
                x=clusters,
                y=outliers_by_cluster_df[col_key],
                marker_color=color_code
            ))

    fig.update_layout(
        barmode='stack',
        title="<b>Risk & Anomaly Concentration by Cluster</b><br><span style='font-size:12px; color:#64748B;'>Stacked count of fuzzy risk severity levels within each cluster cohort</span>",
        template=THEME_TEMPLATE,
        margin=dict(l=40, r=40, t=65, b=40),
        height=320,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        xaxis=dict(title="Cluster Cohort", showgrid=False),
        yaxis=dict(title="Anomalous Records Count", showgrid=True, gridcolor='#F1F5F9'),
        font=dict(family="Inter, sans-serif", size=11, color="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


