"""
DATA SCIENCE VISUALIZATION CHEAT SHEET (Python)
Covers Matplotlib (pyplot), Seaborn, Plotly Express, and general plotting patterns.
Designed as a runnable, commented reference. Uncomment blocks to run examples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set global styles (Matplotlib + Seaborn)
plt.style.use('seaborn-v0_8-whitegrid')  # clean grid background
sns.set_theme(style='whitegrid', palette='deep')

# Sample data for examples
np.random.seed(42)
n = 100
x = np.linspace(0, 10, n)
noise = np.random.normal(scale=1.0, size=n)
y = 2.5 * x + 5 + noise
cat = np.random.choice(['A', 'B', 'C'], size=n)
df = pd.DataFrame({'x': x, 'y': y, 'cat': cat, 'noise': noise})

# ============================================================================
# 1. MATPLOTLIB (PYLOT) BASICS
# ============================================================================

# Core idea: pyplot is stateful; you can also use OO API with Figure/Axes.

# --- Line Plot ---
def mpl_line_plot():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['x'], df['y'], label='y vs x', color='steelblue', linewidth=2)
    ax.set_title('Line Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    fig.tight_layout()
    # plt.show()

# --- Scatter Plot ---
def mpl_scatter_plot():
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(df['x'], df['y'], c=df['noise'], cmap='viridis', alpha=0.7)
    ax.set_title('Scatter Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(sc, ax=ax, label='noise')
    fig.tight_layout()
    # plt.show()

# --- Bar Plot (categorical) ---
def mpl_bar_plot():
    counts = df['cat'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color='coral')
    ax.set_title('Category Counts')
    ax.set_xlabel('category')
    ax.set_ylabel('count')
    fig.tight_layout()
    # plt.show()

# --- Histogram & Density ---
def mpl_hist_kde():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df['noise'], bins=20, color='slateblue', alpha=0.6, density=True, label='hist')
    sns.kdeplot(df['noise'], ax=ax, color='darkred', label='kde')
    ax.set_title('Histogram + KDE')
    ax.legend()
    fig.tight_layout()
    # plt.show()

# --- Subplots grid ---
def mpl_subplots_grid():
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.ravel()
    axes[0].plot(df['x'], df['y'])
    axes[0].set_title('Line')
    axes[1].scatter(df['x'], df['y'])
    axes[1].set_title('Scatter')
    axes[2].hist(df['noise'], bins=15)
    axes[2].set_title('Hist')
    axes[3].bar(df['cat'].value_counts().index, df['cat'].value_counts().values)
    axes[3].set_title('Bar')
    fig.suptitle('Subplots Grid', fontsize=14)
    fig.tight_layout()
    # plt.show()

# --- Twin Axes (two y-scales) ---
def mpl_twin_axes():
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()  # shares x
    ax1.plot(df['x'], df['y'], color='teal', label='y')
    ax2.plot(df['x'], df['noise'], color='orange', label='noise')
    ax1.set_xlabel('x'); ax1.set_ylabel('y', color='teal')
    ax2.set_ylabel('noise', color='orange')
    fig.tight_layout()
    # plt.show()

# --- Save Figure ---
# fig.savefig('plot.png', dpi=300, bbox_inches='tight')

# ============================================================================
# 2. SEABORN (HIGH-LEVEL STATISTICAL PLOTS)
# ============================================================================

# Seaborn works best with tidy DataFrames (columns = variables, rows = observations).

# --- Relational plots ---
def sns_relplot_scatter():
    sns.relplot(data=df, x='x', y='y', hue='cat', size='noise', palette='viridis')
    plt.title('Seaborn relplot (scatter with hue/size)')
    # plt.show()

# --- Categorical plots ---
def sns_catplot_bar():
    sns.catplot(data=df, x='cat', y='y', kind='bar', estimator=np.mean, errorbar='sd')
    plt.title('Mean y by category with SD bars')
    # plt.show()

def sns_box_violin():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(data=df, x='cat', y='y', ax=axes[0])
    axes[0].set_title('Boxplot')
    sns.violinplot(data=df, x='cat', y='y', ax=axes[1], inner='quartile')
    axes[1].set_title('Violin')
    fig.tight_layout()
    # plt.show()

# --- Distributions ---
def sns_distplots():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df['noise'], kde=True, bins=20, ax=axes[0])
    axes[0].set_title('Hist + KDE')
    sns.ecdfplot(df['noise'], ax=axes[1])
    axes[1].set_title('ECDF')
    fig.tight_layout()
    # plt.show()

# --- Joint and pair plots ---
def sns_joint_pair():
    sns.jointplot(data=df, x='x', y='y', kind='hex', cmap='viridis')
    sns.pairplot(df[['x', 'y', 'noise', 'cat']], hue='cat')
    # plt.show()

# --- Heatmaps (correlation) ---
def sns_heatmap_corr():
    corr = df[['x', 'y', 'noise']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    # plt.show()

# --- Faceting (small multiples) ---
def sns_facets():
    g = sns.FacetGrid(df, col='cat')
    g.map_dataframe(sns.scatterplot, x='x', y='y', hue='cat', legend=False)
    g.add_legend()
    # plt.show()

# --- Styling tips ---
# sns.set_theme(style='whitegrid', palette='deep', font_scale=1.1)
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)

# ============================================================================
# 3. PLOTLY EXPRESS (INTERACTIVE)
# ============================================================================

# Plotly Express provides concise interactive plots; works well in notebooks.

def px_scatter():
    fig = px.scatter(df, x='x', y='y', color='cat', size='noise', title='Plotly Scatter')
    # fig.show()
    return fig

def px_line():
    fig = px.line(df, x='x', y='y', color='cat', title='Plotly Line')
    # fig.show()
    return fig

def px_hist():
    fig = px.histogram(df, x='noise', nbins=20, color='cat', marginal='box', title='Plotly Hist + Box')
    # fig.show()
    return fig

def px_heatmap():
    corr = df[['x', 'y', 'noise']].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', title='Correlation Heatmap')
    # fig.show()
    return fig

# ============================================================================
# 4. GENERAL PATTERNS & TIPS
# ============================================================================

# - Keep data tidy: columns = variables, rows = observations.
# - Label axes and add titles; use legends wisely; annotate key points if needed.
# - Use color palettes consistently; be mindful of colorblind-safe palettes.
# - Prefer faceting over overplotting: small multiples show subgroups clearly.
# - For large scatter, use alpha<1 and smaller markers; consider hexbin/kde.
# - Use tight_layout() or constrained_layout=True to reduce overlap.
# - Save with high dpi for print: fig.savefig('plot.png', dpi=300).
# - Interactive needs: Plotly or mplcursors (Matplotlib) for hover info.
# - Reuse style: set_theme / plt.style.use for consistent appearance.

# ============================================================================
# 5. QUICK RECIPE COLLECTION
# ============================================================================

# --- Annotate points ---
def annotate_example():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df['x'], df['y'], alpha=0.6)
    # Annotate max y point
    idx = df['y'].idxmax()
    ax.annotate('max y', xy=(df.loc[idx, 'x'], df.loc[idx, 'y']),
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='black'))
    fig.tight_layout()
    # plt.show()

# --- Horizontal / vertical reference lines ---
def ref_lines():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['x'], df['y'])
    ax.axhline(df['y'].mean(), color='red', linestyle='--', label='mean y')
    ax.axvline(df['x'].median(), color='green', linestyle=':', label='median x')
    ax.legend()
    fig.tight_layout()
    # plt.show()

# --- Customizing ticks ---
def ticks_custom():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['x'], df['y'])
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(int(df['y'].min()), int(df['y'].max()) + 1, 5))
    ax.tick_params(axis='x', rotation=0)
    fig.tight_layout()
    # plt.show()

# --- Log scales ---
def log_scale():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['x'], np.abs(df['noise']) + 1)
    ax.set_yscale('log')
    ax.set_title('Log Y Scale')
    fig.tight_layout()
    # plt.show()

# --- Multiple figures saving loop example ---
def save_multiple():
    for cat_val, sub in df.groupby('cat'):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(sub['x'], sub['y'], alpha=0.7)
        ax.set_title(f'Scatter for {cat_val}')
        fig.tight_layout()
        # fig.savefig(f'scatter_{cat_val}.png', dpi=200)
        plt.close(fig)

# ============================================================================
# 6. TIME SERIES VISUALIZATION
# ============================================================================

# Time series data often needs special date handling and rolling/resampling plots.

def timeseries_basic():
    """Date formatting, resampling, rolling windows"""
    import matplotlib.dates as mdates
    
    # Create time series data
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    ts_data = pd.DataFrame({
        'date': dates,
        'value': np.cumsum(np.random.randn(365)) + 100,
        'category': np.random.choice(['A', 'B'], 365)
    })
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Basic time series line
    axes[0].plot(ts_data['date'], ts_data['value'], linewidth=1)
    axes[0].set_title('Time Series - Daily Values')
    axes[0].set_ylabel('Value')
    
    # Rolling mean (30-day window)
    rolling = ts_data.set_index('date')['value'].rolling(window=30).mean()
    axes[1].plot(ts_data['date'], ts_data['value'], alpha=0.3, label='Daily')
    axes[1].plot(rolling.index, rolling.values, color='red', linewidth=2, label='30-day MA')
    axes[1].set_title('Time Series with Rolling Mean')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    
    # Monthly resampling
    monthly = ts_data.set_index('date')['value'].resample('M').mean()
    axes[2].bar(monthly.index, monthly.values, width=20, alpha=0.7)
    axes[2].set_title('Monthly Average Values')
    axes[2].set_ylabel('Value')
    
    # Format x-axis dates
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    fig.tight_layout()
    # plt.show()

def timeseries_area_stacked():
    """Stacked area chart for composition over time"""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = {
        'date': dates,
        'A': np.random.rand(100).cumsum(),
        'B': np.random.rand(100).cumsum(),
        'C': np.random.rand(100).cumsum()
    }
    df_ts = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(df_ts['date'], df_ts['A'], df_ts['B'], df_ts['C'],
                 labels=['A', 'B', 'C'], alpha=0.7)
    ax.set_title('Stacked Area Chart - Composition Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend(loc='upper left')
    fig.tight_layout()
    # plt.show()

# ============================================================================
# 7. STATISTICAL VISUALIZATION
# ============================================================================

def statistical_regression():
    """Regression line with confidence interval"""
    # Seaborn makes this easy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Seaborn regplot (includes CI)
    sns.regplot(data=df, x='x', y='y', ax=axes[0], scatter_kws={'alpha': 0.5})
    axes[0].set_title('Regression with 95% CI (Seaborn)')
    
    # Seaborn lmplot alternative (FacetGrid)
    sns.regplot(data=df, x='x', y='y', ax=axes[1], order=2, scatter_kws={'alpha': 0.5})
    axes[1].set_title('Polynomial Regression (order=2)')
    
    fig.tight_layout()
    # plt.show()

def statistical_residuals():
    """Residual plot for regression diagnostics"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Residual plot
    sns.residplot(data=df, x='x', y='y', ax=axes[0], lowess=True, color='steelblue')
    axes[0].set_title('Residual Plot')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
    
    # Q-Q plot for normality check
    from scipy import stats
    stats.probplot(df['noise'], dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    
    fig.tight_layout()
    # plt.show()

def statistical_error_bars():
    """Error bars showing uncertainty"""
    # Group data and compute stats
    grouped = df.groupby('cat')['y'].agg(['mean', 'std', 'sem'])
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Error bars with std
    axes[0].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                     fmt='o', capsize=5, capthick=2, markersize=8)
    axes[0].set_title('Mean ± Std Dev')
    axes[0].set_ylabel('Value')
    
    # Error bars with SEM
    axes[1].errorbar(grouped.index, grouped['mean'], yerr=grouped['sem'],
                     fmt='s', capsize=5, capthick=2, markersize=8, color='coral')
    axes[1].set_title('Mean ± SEM')
    axes[1].set_ylabel('Value')
    
    # Bar chart with error bars
    axes[2].bar(grouped.index, grouped['mean'], yerr=grouped['std'],
                capsize=5, alpha=0.7, color='steelblue')
    axes[2].set_title('Bar Chart with Error Bars')
    axes[2].set_ylabel('Value')
    
    fig.tight_layout()
    # plt.show()

def statistical_swarm_strip():
    """Swarm and strip plots for categorical data"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Strip plot - jittered scatter
    sns.stripplot(data=df, x='cat', y='y', ax=axes[0], alpha=0.6, jitter=True)
    axes[0].set_title('Strip Plot (Jittered Points)')
    
    # Swarm plot - non-overlapping points (better for smaller datasets)
    sns.swarmplot(data=df, x='cat', y='y', ax=axes[1], alpha=0.7)
    axes[1].set_title('Swarm Plot (Non-overlapping)')
    
    fig.tight_layout()
    # plt.show()

# ============================================================================
# 8. CONTOUR AND 3D PLOTS
# ============================================================================

def contour_plots():
    """Contour plots for 2D functions/densities"""
    # Create mesh grid
    x_grid = np.linspace(-3, 3, 100)
    y_grid = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Contour lines
    cs = axes[0].contour(X, Y, Z, levels=10, cmap='viridis')
    axes[0].clabel(cs, inline=True, fontsize=8)
    axes[0].set_title('Contour Lines')
    
    # Filled contour
    cf = axes[1].contourf(X, Y, Z, levels=20, cmap='viridis')
    fig.colorbar(cf, ax=axes[1])
    axes[1].set_title('Filled Contour')
    
    # Contour with scatter overlay
    axes[2].contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    axes[2].scatter(df['x'][:50] - 5, df['noise'][:50], c='red', s=20, alpha=0.8)
    axes[2].set_title('Contour + Scatter')
    
    fig.tight_layout()
    # plt.show()

def plot_3d():
    """3D surface and scatter plots"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 5))
    
    # 3D scatter
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(df['x'], df['y'], df['noise'], c=df['noise'], cmap='plasma', alpha=0.6)
    ax1.set_title('3D Scatter')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Noise')
    
    # 3D surface
    ax2 = fig.add_subplot(132, projection='3d')
    x_grid = np.linspace(-3, 3, 50)
    y_grid = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    surf = ax2.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
    ax2.set_title('3D Surface')
    fig.colorbar(surf, ax=ax2, shrink=0.5)
    
    # 3D wireframe
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_wireframe(X, Y, Z, color='steelblue', alpha=0.5)
    ax3.set_title('3D Wireframe')
    
    fig.tight_layout()
    # plt.show()

# ============================================================================
# 9. PIE CHARTS & COMPOSITION
# ============================================================================

def pie_donut_charts():
    """Pie and donut charts (use sparingly - bars often better)"""
    counts = df['cat'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Basic pie chart
    axes[0].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette('pastel'))
    axes[0].set_title('Pie Chart')
    
    # Donut chart (pie with hole)
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette('pastel'),
                wedgeprops={'width': 0.4})
    axes[1].set_title('Donut Chart')
    
    fig.tight_layout()
    # plt.show()

# ============================================================================
# 10. GEOGRAPHIC MAPS
# ============================================================================

def geographic_maps():
    """Basic choropleth and scatter maps using Plotly"""
    # Simple US state example (requires plotly)
    state_data = pd.DataFrame({
        'state': ['CA', 'TX', 'FL', 'NY', 'PA'],
        'value': [100, 85, 70, 90, 60]
    })
    
    # Choropleth map
    fig1 = px.choropleth(
        state_data,
        locations='state',
        locationmode='USA-states',
        color='value',
        scope='usa',
        title='Choropleth Map - US States'
    )
    # fig1.show()
    
    # Scatter map (geographic coordinates)
    city_data = pd.DataFrame({
        'city': ['New York', 'Los Angeles', 'Chicago', 'Houston'],
        'lat': [40.7128, 34.0522, 41.8781, 29.7604],
        'lon': [-74.0060, -118.2437, -87.6298, -95.3698],
        'population': [8.4, 3.9, 2.7, 2.3]
    })
    
    fig2 = px.scatter_geo(
        city_data,
        lat='lat',
        lon='lon',
        size='population',
        hover_name='city',
        scope='usa',
        title='Scatter Map - US Cities'
    )
    # fig2.show()
    
    return fig1, fig2

# ============================================================================
# 11. ADVANCED LAYOUTS & CUSTOMIZATION
# ============================================================================

def advanced_subplots_gridspec():
    """GridSpec for complex subplot arrangements"""
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)
    
    # Large plot spanning multiple cells
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.plot(df['x'], df['y'], 'o-', alpha=0.6)
    ax1.set_title('Main Plot (2x2)')
    
    # Smaller plots
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(df['noise'], bins=15, color='coral', alpha=0.7)
    ax2.set_title('Hist')
    
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.boxplot([df[df['cat'] == c]['y'].values for c in ['A', 'B', 'C']])
    ax3.set_title('Box')
    
    ax4 = fig.add_subplot(gs[2, :])
    ax4.scatter(df['x'], df['y'], c=df['noise'], cmap='viridis', alpha=0.6)
    ax4.set_title('Bottom Span (1x3)')
    
    # plt.show()

def inset_axes_example():
    """Inset axes (plot within a plot)"""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['x'], df['y'], 'o-', alpha=0.6)
    ax.set_title('Main Plot with Inset')
    
    # Create inset
    ax_inset = inset_axes(ax, width="40%", height="30%", loc='upper right')
    ax_inset.hist(df['noise'], bins=15, color='steelblue', alpha=0.7)
    ax_inset.set_title('Inset Hist', fontsize=10)
    
    fig.tight_layout()
    # plt.show()

def custom_legends_colorbars():
    """Custom legend positioning and colorbar tweaks"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Custom legend position
    for cat_val in df['cat'].unique():
        subset = df[df['cat'] == cat_val]
        axes[0].scatter(subset['x'], subset['y'], label=f'Cat {cat_val}', alpha=0.7, s=50)
    
    # Legend outside plot area
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    axes[0].set_title('Legend Outside')
    
    # Custom colorbar
    sc = axes[1].scatter(df['x'], df['y'], c=df['noise'], cmap='coolwarm', s=50)
    cbar = fig.colorbar(sc, ax=axes[1])
    cbar.set_label('Noise Level', rotation=270, labelpad=20)
    axes[1].set_title('Custom Colorbar')
    
    fig.tight_layout()
    # plt.show()

# ============================================================================
# 12. ANIMATION BASICS
# ============================================================================

def animation_simple():
    """Basic animation using FuncAnimation"""
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=(6, 4))
    xdata, ydata = [], []
    ln, = ax.plot([], [], 'o-', animated=True)
    
    def init():
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 30)
        return ln,
    
    def update(frame):
        xdata.append(frame)
        ydata.append(frame**0.5 * 5 + np.random.randn())
        ln.set_data(xdata, ydata)
        return ln,
    
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 10, 50),
                       init_func=init, blit=True, interval=50)
    
    # To save: ani.save('animation.gif', writer='pillow', fps=20)
    # plt.show()
    return ani

# ============================================================================
# 13. SEABORN ADVANCED PATTERNS
# ============================================================================

def seaborn_advanced():
    """More Seaborn features: count plots, point plots, bar plots with CIs"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Count plot (frequency)
    sns.countplot(data=df, x='cat', ax=axes[0, 0], palette='Set2')
    axes[0, 0].set_title('Count Plot')
    
    # Point plot (means with CI)
    sns.pointplot(data=df, x='cat', y='y', ax=axes[0, 1], errorbar='sd', capsize=0.1)
    axes[0, 1].set_title('Point Plot with SD')
    
    # Bar plot with custom estimator
    sns.barplot(data=df, x='cat', y='y', estimator=np.median, errorbar=('pi', 75),
                ax=axes[1, 0], palette='muted')
    axes[1, 0].set_title('Bar Plot - Median with 75% PI')
    
    # Categorical scatter with box overlay
    sns.boxplot(data=df, x='cat', y='y', ax=axes[1, 1], width=0.3, palette='pastel')
    sns.stripplot(data=df, x='cat', y='y', ax=axes[1, 1], color='black', alpha=0.3, size=3)
    axes[1, 1].set_title('Box + Strip Overlay')
    
    fig.tight_layout()
    # plt.show()

# ============================================================================
# 14. COLOR & STYLE NOTES
# ============================================================================

# Matplotlib built-ins: plt.cm.viridis, plasma, magma, cividis, coolwarm, etc.
# Seaborn palettes: sns.color_palette('deep'), 'muted', 'bright', 'dark', 'colorblind'
# Diverging palettes for centered data: 'RdBu_r', 'coolwarm'
# Qualitative for categories: 'tab10', 'Set2', 'Pastel1'

# ============================================================================
# MAIN DEMO (lightweight sanity checks)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COMPREHENSIVE DATA SCIENCE VISUALIZATION CHEAT SHEET")
    print("=" * 70)
    print("\nUncomment plt.show() or fig.show() lines to view plots.")
    print("\nSections covered:")
    print("  1. Matplotlib basics")
    print("  2. Seaborn statistical plots")
    print("  3. Plotly interactive plots")
    print("  4. General patterns & customization")
    print("  5. Quick recipes")
    print("  6. Time series visualization")
    print("  7. Statistical visualization (regression, residuals, error bars)")
    print("  8. Contour and 3D plots")
    print("  9. Pie charts & composition")
    print(" 10. Geographic maps")
    print(" 11. Advanced layouts (GridSpec, insets, legends)")
    print(" 12. Animation basics")
    print(" 13. Seaborn advanced patterns")
    print("=" * 70)
    
    # Run examples (lightweight checks)
    mpl_line_plot()
    mpl_scatter_plot()
    mpl_bar_plot()
    mpl_hist_kde()
    mpl_subplots_grid()
    mpl_twin_axes()
    sns_relplot_scatter()
    sns_catplot_bar()
    sns_box_violin()
    sns_distplots()
    sns_joint_pair()
    sns_heatmap_corr()
    sns_facets()
    px_scatter(); px_line(); px_hist(); px_heatmap()
    annotate_example(); ref_lines(); ticks_custom(); log_scale(); save_multiple()
    
    # New sections
    timeseries_basic()
    timeseries_area_stacked()
    statistical_regression()
    statistical_residuals()
    statistical_error_bars()
    statistical_swarm_strip()
    contour_plots()
    plot_3d()
    pie_donut_charts()
    geographic_maps()
    advanced_subplots_gridspec()
    inset_axes_example()
    custom_legends_colorbars()
    animation_simple()
    seaborn_advanced()
    
    print("\nAll visualization examples loaded successfully!")
    print("Explore each function to learn specific patterns.")
