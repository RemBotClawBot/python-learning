#!/usr/bin/env python3
"""
Data Visualization Examples
==========================

This module demonstrates Python data visualization techniques using:
- Matplotlib (basic plotting)
- Seaborn (statistical visualization)
- Plotly (interactive plots)
- Pandas integration
- Export options (PNG, PDF, HTML)

Covers: line charts, bar charts, histograms, scatter plots, heatmaps,
subplots, custom styling, and interactive dashboards.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not installed. Run: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Note: seaborn not installed. Run: pip install seaborn")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: plotly not installed. Run: pip install plotly")


# ---------- Helper Functions ----------

def generate_sample_data() -> pd.DataFrame:
    """Generate sample data for visualization examples."""
    np.random.seed(42)
    
    # Generate time series data
    dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
    n_days = len(dates)
    
    # Create DataFrame with multiple metrics
    data = pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(loc=1000, scale=200, size=n_days).cumsum() + 10000,
        'temperature': np.random.normal(loc=20, scale=5, size=n_days),
        'rainfall': np.random.exponential(scale=5, size=n_days),
        'visitors': np.random.poisson(lam=500, size=n_days),
        'revenue': np.random.lognormal(mean=8, sigma=0.5, size=n_days),
    })
    
    # Add categorical data
    categories = ['A', 'B', 'C', 'D', 'E']
    data['category'] = np.random.choice(categories, size=n_days, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    
    # Add day of week
    data['day_of_week'] = data['date'].dt.day_name()
    data['month'] = data['date'].dt.month_name()
    
    # Calculate moving averages
    data['sales_ma7'] = data['sales'].rolling(window=7).mean()
    data['sales_ma30'] = data['sales'].rolling(window=30).mean()
    
    # Add some outliers
    outlier_indices = np.random.choice(n_days, size=10, replace=False)
    data.loc[outlier_indices, 'sales'] *= 1.5
    data.loc[outlier_indices, 'revenue'] *= 2.0
    
    return data


def setup_plot_style():
    """Set up consistent plot styling."""
    if MATPLOTLIB_AVAILABLE:
        # Set style
        plt.style.use('default')
        
        # Set rcParams for better defaults
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
        
        # Set Seaborn style if available
        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_palette("husl")


# ---------- Matplotlib Examples ----------

def matplotlib_line_chart(data: pd.DataFrame) -> None:
    """Create a line chart with multiple series."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    print("\n" + "="*60)
    print("Matplotlib: Line Chart Example")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Sales over time
    ax1 = axes[0, 0]
    ax1.plot(data['date'], data['sales'], label='Daily Sales', color='blue', alpha=0.7)
    ax1.plot(data['date'], data['sales_ma7'], label='7-Day MA', color='red', linewidth=2)
    ax1.plot(data['date'], data['sales_ma30'], label='30-Day MA', color='green', linewidth=2)
    ax1.set_title('Sales Over Time with Moving Averages')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # Plot 2: Temperature histogram
    ax2 = axes[0, 1]
    ax2.hist(data['temperature'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_title('Temperature Distribution')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Frequency')
    ax2.axvline(data['temperature'].mean(), color='red', linestyle='--', 
                label=f'Mean: {data["temperature"].mean():.1f}°C')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot
    ax3 = axes[1, 0]
    scatter = ax3.scatter(data['temperature'], data['sales'], 
                          c=data['visitors'], s=data['revenue']/100, 
                          alpha=0.6, cmap='viridis')
    ax3.set_title('Sales vs Temperature (size=revenue, color=visitors)')
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Sales')
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Visitors')
    
    # Plot 4: Bar chart by month
    ax4 = axes[1, 1]
    monthly_sales = data.groupby('month')['sales'].mean()
    months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']
    monthly_sales = monthly_sales.reindex(months_order, fill_value=0)
    
    bars = ax4.bar(range(len(monthly_sales)), monthly_sales.values, 
                   color=cm.viridis(np.linspace(0, 1, len(monthly_sales))))
    ax4.set_title('Average Sales by Month')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Average Sales')
    ax4.set_xticks(range(len(monthly_sales)))
    ax4.set_xticklabels([m[:3] for m in monthly_sales.index], rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(monthly_sales.values):
        ax4.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "matplotlib_line_chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved line chart to: {output_path}")
    
    plt.show()


def matplotlib_subplot_example(data: pd.DataFrame) -> None:
    """Create a multi-panel dashboard with subplots."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    print("\n" + "="*60)
    print("Matplotlib: Subplot Dashboard Example")
    print("="*60)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3)
    
    # Plot 1: Sales time series (spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(data['date'], data['sales'], 'b-', alpha=0.7, label='Daily')
    ax1.fill_between(data['date'], data['sales'].min(), data['sales'], alpha=0.2)
    ax1.set_title('Sales Timeline', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sales ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Category distribution (pie chart)
    ax2 = fig.add_subplot(gs[0, 2])
    category_counts = data['category'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
    wedges, texts, autotexts = ax2.pie(category_counts.values, labels=category_counts.index,
                                          autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Category Distribution', fontsize=14, fontweight='bold')
    
    # Plot 3: Temperature vs Visitors scatter
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(data['temperature'], data['visitors'], 
                          c=data['revenue'], cmap='plasma', alpha=0.7)
    ax3.set_title('Temperature vs Visitors')
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Visitors')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Revenue')
    
    # Plot 4: Revenue histogram
    ax4 = fig.add_subplot(gs[1, 1:])
    n_bins = 30
    counts, bins, patches = ax4.hist(data['revenue'], bins=n_bins, 
                                       color='steelblue', edgecolor='black', alpha=0.7)
    ax4.set_title('Revenue Distribution')
    ax4.set_xlabel('Revenue ($)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # Highlight bins above mean
    mean_revenue = data['revenue'].mean()
    for patch, bin_edge in zip(patches, bins[:-1]):
        if bin_edge > mean_revenue:
            patch.set_facecolor('darkred')
    
    # Plot 5: Box plot by day of week
    ax5 = fig.add_subplot(gs[2, :])
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    box_data = [data[data['day_of_week'] == day]['sales'].values for day in day_order]
    
    box = ax5.boxplot(box_data, labels=day_order, patch_artist=True)
    ax5.set_title('Sales Distribution by Day of Week', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Sales ($)')
    ax5.grid(True, alpha=0.3)
    
    # Color boxes
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(day_order)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.suptitle('Data Analysis Dashboard', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join("reports", "matplotlib_dashboard.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved dashboard to: {output_path}")
    
    plt.show()


# ---------- Seaborn Examples ----------

def seaborn_examples(data: pd.DataFrame) -> None:
    """Create statistical visualizations with Seaborn."""
    if not SEABORN_AVAILABLE:
        return
    
    print("\n" + "="*60)
    print("Seaborn: Statistical Visualization Examples")
    print("="*60)
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Heatmap correlation
    ax1 = axes[0, 0]
    numeric_cols = ['sales', 'temperature', 'rainfall', 'visitors', 'revenue']
    correlation_matrix = data[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Correlation Heatmap')
    
    # Plot 2: Violin plot by category
    ax2 = axes[0, 1]
    sns.violinplot(data=data, x='category', y='sales', ax=ax2, inner='quartile')
    ax2.set_title('Sales Distribution by Category')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Sales')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Pairplot (subset)
    ax3 = axes[0, 2]
    # For pairplot, we need a different approach since it creates its own figure
    # Instead, let's do a scatter plot with regression line
    sns.regplot(data=data.sample(200), x='temperature', y='visitors', 
                scatter_kws={'alpha': 0.5, 's': 20}, line_kws={'color': 'red'}, ax=ax3)
    ax3.set_title('Temperature vs Visitors (with Regression)')
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Visitors')
    
    # Plot 4: Distribution with KDE
    ax4 = axes[1, 0]
    sns.histplot(data=data, x='revenue', kde=True, bins=30, ax=ax4)
    ax4.set_title('Revenue Distribution with KDE')
    ax4.set_xlabel('Revenue ($)')
    ax4.set_ylabel('Density')
    
    # Plot 5: Box plot with swarm
    ax5 = axes[1, 1]
    # Sample data for swarm plot (too many points for full dataset)
    sample_data = data.groupby('category').apply(lambda x: x.sample(min(10, len(x)))).reset_index(drop=True)
    sns.boxplot(data=data, x='category', y='revenue', ax=ax5, color='lightgray')
    sns.swarmplot(data=sample_data, x='category', y='revenue', ax=ax5, 
                  size=4, palette='dark:black', alpha=0.7)
    ax5.set_title('Revenue by Category (with Swarm)')
    ax5.set_xlabel('Category')
    ax5.set_ylabel('Revenue ($)')
    ax5.tick_params(axis='x', rotation=45)
    
    # Plot 6: Time series with confidence intervals
    ax6 = axes[1, 2]
    weekly_data = data.resample('W', on='date').agg({
        'sales': 'mean',
        'temperature': 'mean',
        'visitors': 'sum'
    }).reset_index()
    
    sns.lineplot(data=weekly_data, x='date', y='sales', ax=ax6, label='Sales')
    ax6.fill_between(weekly_data['date'], 
                     weekly_data['sales'] - weekly_data['sales'].std()/2,
                     weekly_data['sales'] + weekly_data['sales'].std()/2,
                     alpha=0.2)
    ax6.set_title('Weekly Sales with Confidence Band')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Average Sales')
    ax6.legend()
    
    plt.suptitle('Seaborn Statistical Visualizations', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join("reports", "seaborn_examples.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Seaborn visualizations to: {output_path}")
    
    plt.show()
    
    # Create a pairplot separately (creates its own figure)
    print("\nCreating pairplot...")
    pairplot_data = data[numeric_cols + ['category']].sample(500)  # Sample for performance
    pairplot = sns.pairplot(pairplot_data, hue='category', diag_kind='kde', 
                            plot_kws={'alpha': 0.6, 's': 20})
    pairplot.fig.suptitle('Pairplot: Relationships Between Variables', y=1.02)
    
    pairplot_path = os.path.join("reports", "seaborn_pairplot.png")
    pairplot.savefig(pairplot_path, dpi=300, bbox_inches='tight')
    print(f"Saved pairplot to: {pairplot_path}")


# ---------- Plotly Examples ----------

def plotly_examples(data: pd.DataFrame) -> None:
    """Create interactive visualizations with Plotly."""
    if not PLOTLY_AVAILABLE:
        return
    
    print("\n" + "="*60)
    print("Plotly: Interactive Visualization Examples")
    print("="*60)
    
    # Example 1: Interactive time series with range selector
    fig1 = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Sales Over Time', 'Daily Visitors'),
        vertical_spacing=0.15
    )
    
    # Sales trace
    fig1.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['sales'],
            mode='lines',
            name='Sales',
            line=dict(color='royalblue', width=2),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Sales: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Moving average trace
    fig1.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['sales_ma30'],
            mode='lines',
            name='30-Day MA',
            line=dict(color='firebrick', width=2, dash='dash'),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>30-Day MA: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Visitors trace
    fig1.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['visitors'],
            mode='lines+markers',
            name='Visitors',
            line=dict(color='forestgreen', width=1),
            marker=dict(size=4),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Visitors: %{y:.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig1.update_layout(
        title='Interactive Time Series Dashboard',
        height=800,
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        xaxis2=dict(
            rangeslider=dict(visible=False),
            type="date"
        )
    )
    
    # Save as HTML
    fig1_path = os.path.join("reports", "plotly_timeseries.html")
    fig1.write_html(fig1_path)
    print(f"Saved interactive time series to: {fig1_path}")
    
    # Example 2: Scatter plot matrix
    fig2 = px.scatter_matrix(
        data.sample(300),  # Sample for performance
        dimensions=['sales', 'temperature', 'visitors', 'revenue'],
        color='category',
        title='Scatter Plot Matrix by Category',
        height=700
    )
    
    fig2.update_traces(diagonal_visible=False)
    fig2.update_layout(
        dragmode='select',
        hovermode='closest'
    )
    
    fig2_path = os.path.join("reports", "plotly_scatter_matrix.html")
    fig2.write_html(fig2_path)
    print(f"Saved scatter matrix to: {fig2_path}")
    
    # Example 3: 3D scatter plot
    fig3 = px.scatter_3d(
        data.sample(200),
        x='temperature',
        y='visitors',
        z='sales',
        color='revenue',
        size='rainfall',
        hover_data=['date', 'category'],
        title='3D Scatter: Temperature vs Visitors vs Sales',
        labels={'temperature': 'Temperature (°C)', 'visitors': 'Visitors', 'sales': 'Sales'},
        height=700
    )
    
    fig3_path = os.path.join("reports", "plotly_3d_scatter.html")
    fig3.write_html(fig3_path)
    print(f"Saved 3D scatter plot to: {fig2_path}")
    
    # Example 4: Sunburst chart
    monthly_data = data.groupby(['month', 'category']).agg({
        'sales': 'sum',
        'revenue': 'mean',
        'visitors': 'sum'
    }).reset_index()
    
    fig4 = px.sunburst(
        monthly_data,
        path=['month', 'category'],
        values='sales',
        color='revenue',
        hover_data=['visitors'],
        title='Sales Hierarchy: Month → Category',
        height=700
    )
    
    fig4_path = os.path.join("reports", "plotly_sunburst.html")
    fig4.write_html(fig4_path)
    print(f"Saved sunburst chart to: {fig4_path}")
    
    # Example 5: Animated bubble chart
    data['month_num'] = data['date'].dt.month
    monthly_agg = data.groupby(['month_num', 'category']).agg({
        'sales': 'mean',
        'temperature': 'mean',
        'visitors': 'sum',
        'revenue': 'mean'
    }).reset_index()
    
    monthly_agg['month_name'] = monthly_agg['month_num'].apply(lambda x: datetime(2025, x, 1).strftime('%B'))
    
    fig5 = px.scatter(
        monthly_agg,
        x='temperature',
        y='visitors',
        size='sales',
        color='category',
        hover_name='category',
        animation_frame='month_name',
        animation_group='category',
        size_max=60,
        title='Monthly Metrics Animation',
        labels={'temperature': 'Avg Temperature', 'visitors': 'Total Visitors'},
        height=600
    )
    
    fig5_path = os.path.join("reports", "plotly_animation.html")
    fig5.write_html(fig5_path)
    print(f"Saved animated bubble chart to: {fig5_path}")
    
    print("\nInteractive Plotly charts saved as HTML files.")
    print("Open them in a web browser to interact with the visualizations!")
    
    return fig1, fig2, fig3, fig4, fig5


def export_to_pdf_report(data: pd.DataFrame) -> None:
    """Create a PDF report with multiple visualizations."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        
        print("\n" + "="*60)
        print("Exporting PDF Report")
        print("="*60)
        
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, "data_visualization_report.pdf")
        
        with PdfPages(pdf_path) as pdf:
            # Summary statistics page
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # Calculate summary statistics
            summary_data = {
                'Metric': ['Total Sales', 'Average Daily Sales', 'Max Sales', 'Min Sales',
                          'Total Revenue', 'Average Temperature', 'Total Visitors',
                          'Records Count', 'Date Range'],
                'Value': [
                    f"${data['sales'].sum():,.0f}",
                    f"${data['sales'].mean():,.0f}",
                    f"${data['sales'].max():,.0f}",
                    f"${data['sales'].min():,.0f}",
                    f"${data['revenue'].sum():,.0f}",
                    f"{data['temperature'].mean():.1f}°C",
                    f"{data['visitors'].sum():,.0f}",
                    f"{len(data):,.0f}",
                    f"{data['date'].min().date()} to {data['date'].max().date()}"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            table = ax.table(cellText=summary_df.values,
                            colLabels=summary_df.columns,
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.4, 0.3])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            
            ax.set_title('Data Summary Report', fontsize=16, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Correlation heatmap page
            fig, ax = plt.subplots(figsize=(10, 8))
            numeric_cols = ['sales', 'temperature', 'rainfall', 'visitors', 'revenue']
            corr_matrix = data[numeric_cols].corr()
            
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr_matrix)))
            ax.set_yticks(range(len(corr_matrix)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_matrix.columns)
            
            # Add correlation values
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white")
            
            ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Time series page
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Sales plot
            axes[0].plot(data['date'], data['sales'], 'b-', alpha=0.7, label='Daily Sales')
            axes[0].plot(data['date'], data['sales_ma30'], 'r-', linewidth=2, label='30-Day MA')
            axes[0].set_title('Sales Over Time', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Sales ($)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Temperature plot
            axes[1].plot(data['date'], data['temperature'], 'g-', alpha=0.7)
            axes[1].fill_between(data['date'], 
                                 data['temperature'].min(), 
                                 data['temperature'],
                                 alpha=0.2, color='green')
            axes[1].set_title('Temperature Over Time', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Temperature (°C)')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            print(f"PDF report saved to: {pdf_path}")
            
    except ImportError:
        print("Could not create PDF report: matplotlib.backends.backend_pdf not available")
    except Exception as e:
        print(f"Error creating PDF report: {e}")


# ---------- Main Execution ----------

def main():
    """Run all visualization examples."""
    print("Data Visualization Examples")
    print("="*60)
    print("\nThis module demonstrates data visualization techniques in Python.")
    print("Generating sample data...")
    
    # Generate sample data
    data = generate_sample_data()
    print(f"Generated {len(data):,} records from {data['date'].min().date()} to {data['date'].max().date()}")
    print(f"Columns: {', '.join(data.columns)}")
    
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Setup plot style
    setup_plot_style()
    
    # Run examples based on availability
    if MATPLOTLIB_AVAILABLE:
        matplotlib_line_chart(data)
        matplotlib_subplot_example(data)
        export_to_pdf_report(data)
    else:
        print("\nSkipping Matplotlib examples (matplotlib not installed)")
        print("Install with: pip install matplotlib")
    
    if SEABORN_AVAILABLE:
        seaborn_examples(data)
    else:
        print("\nSkipping Seaborn examples (seaborn not installed)")
        print("Install with: pip install seaborn")
    
    if PLOTLY_AVAILABLE:
        plotly_examples(data)
    else:
        print("\nSkipping Plotly examples (plotly not installed)")
        print("Install with: pip install plotly")
    
    # Save data to CSV for reference
    csv_path = os.path.join("reports", "sample_data.csv")
    data.to_csv(csv_path, index=False)
    print(f"\nSample data saved to: {csv_path}")
    
    # Create requirements file
    requirements = """# Data Visualization Dependencies
# Core plotting
matplotlib>=3.7.0
seaborn>=0.12.0

# Interactive visualizations
plotly>=5.15.0

# Data manipulation
numpy>=1.24.0
pandas>=2.0.0

# Optional (for PDF export)
# No additional packages needed for PDF export with matplotlib
"""
    
    with open("requirements-visualization.txt", "w") as f:
        f.write(requirements)
    print("\nCreated requirements-visualization.txt")
    
    print("\n" + "="*60)
    print("Data Visualization Examples Complete!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - reports/matplotlib_line_chart.png")
    print(f"  - reports/matplotlib_dashboard.png")
    print(f"  - reports/seaborn_examples.png")
    print(f"  - reports/seaborn_pairplot.png")
    print(f"  - reports/plotly_timeseries.html")
    print(f"  - reports/plotly_scatter_matrix.html")
    print(f"  - reports/plotly_3d_scatter.html")
    print(f"  - reports/plotly_sunburst.html")
    print(f"  - reports/plotly_animation.html")
    print(f"  - reports/data_visualization_report.pdf")
    print(f"  - reports/sample_data.csv")
    print(f"  - requirements-visualization.txt")
    print("\nTo install dependencies:")
    print("  pip install -r requirements-visualization.txt")
    print("\nTo run:")
    print("  python data_visualization.py")


if __name__ == "__main__":
    main()