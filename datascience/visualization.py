"""
================================================================================
DATA SCIENCE VISUALIZATION CHEAT SHEET (Python)
================================================================================

COMPREHENSIVE GUIDE covering all major data visualization techniques and tools
for data science, data analysis, and scientific computing.

LIBRARIES COVERED:
- Matplotlib (pyplot + OO API): Core plotting library with full customization
- Seaborn: High-level statistical visualization built on Matplotlib
- Plotly Express: Interactive web-based visualizations

CONTENTS:
1.  Matplotlib Basics: line, scatter, bar, histogram, subplots, twin axes,
    annotations, custom ticks, log scales, saving figures
2.  Seaborn Statistical Plots: relplot, catplot, box/violin, distributions,
    joint/pair plots, heatmaps, faceting, styling
3.  Plotly Interactive: scatter, line, histogram, heatmap with hover tooltips
4.  General Patterns: design principles, best practices, tips
5.  Quick Recipes: common customization patterns (annotations, reference lines)
6.  Time Series: date formatting, rolling windows, resampling, stacked areas
7.  Statistical Visualization: regression with CI, residual plots, Q-Q plots,
    error bars, swarm/strip plots
8.  Contour & 3D: contour plots, 3D scatter/surface/wireframe
9.  Pie Charts: pie and donut charts for composition
10. Geographic Maps: choropleth and scatter maps (Plotly)
11. Advanced Layouts: GridSpec, inset axes, custom legends/colorbars
12. Animation: FuncAnimation basics for dynamic plots
13. Seaborn Advanced: count plots, point plots, custom estimators, overlays
14. Color & Style: colormaps, palettes, themes, colorblind-friendly options
15. Specialized Visualizations: dendrograms (hierarchical clustering), radar/spider
    charts (multivariate comparison), waterfall charts (sequential changes), 
    Sankey diagrams (flow visualization), network graphs (relationships/connections)

USAGE:
- Each section contains runnable, commented examples
- Uncomment plt.show() or fig.show() lines to display plots
- Use as learning reference or quick lookup for syntax
- All examples use sample DataFrame 'df' generated at top

DESIGN FOR:
- Learning visualization fundamentals
- Quick syntax reference for common plot types
- Understanding when to use each visualization type
- Best practices for clear, effective data communication

================================================================================
"""

# ============================================================================
# IMPORTS - Core Libraries for Data Visualization
# ============================================================================

import numpy as np              # Numerical computing: arrays, math operations, random numbers
import pandas as pd             # Data manipulation: DataFrames, Series, data wrangling
import matplotlib.pyplot as plt # Core plotting: figures, axes, basic plots (line, scatter, etc.)
import seaborn as sns           # Statistical visualization: built on Matplotlib, higher-level API
import plotly.express as px     # Interactive plots: web-based visualizations with hover/zoom

# Optional imports (used in specific sections, imported within functions when needed):
# - matplotlib.dates: Date formatting for time series x-axes
# - matplotlib.animation: Creating animated plots (FuncAnimation)
# - matplotlib.gridspec: Advanced subplot layouts (GridSpec)
# - mpl_toolkits.mplot3d: 3D plotting (Axes3D)
# - mpl_toolkits.axes_grid1.inset_locator: Inset axes (plot-in-plot)
# - scipy.stats: Statistical functions (Q-Q plots, probability distributions)
# - scipy.cluster.hierarchy: Hierarchical clustering (dendrograms)
# - networkx: Network/graph visualization
# - plotly.graph_objects: Low-level Plotly for Sankey diagrams

# ============================================================================
# GLOBAL STYLE SETTINGS
# ============================================================================

# Set global styles for consistent appearance across all plots
plt.style.use('seaborn-v0_8-whitegrid')  # Clean grid background
sns.set_theme(style='whitegrid', palette='deep')  # Seaborn defaults

# ============================================================================
# SAMPLE DATA GENERATION
# ============================================================================
# Create synthetic dataset for demonstrating various plot types

np.random.seed(42)  # Reproducible random numbers
n = 100             # Number of samples

# Generate sample data
x = np.linspace(0, 10, n)                    # Evenly spaced x values
noise = np.random.normal(scale=1.0, size=n)  # Random noise
y = 2.5 * x + 5 + noise                      # Linear relationship with noise
cat = np.random.choice(['A', 'B', 'C'], size=n)  # Categorical variable

# Combine into DataFrame (Seaborn and Plotly prefer tidy data)
df = pd.DataFrame({
    'x': x,         # Continuous predictor
    'y': y,         # Continuous response
    'cat': cat,     # Categories
    'noise': noise  # Additional continuous variable
})

# ============================================================================
# 1. MATPLOTLIB (PYLOT) BASICS
# ============================================================================

# Core idea: pyplot is stateful; you can also use OO API with Figure/Axes.

# --- Line Plot ---
def mpl_line_plot():
    """Basic line plot showing trend over continuous x-axis"""
    # Create figure and axis objects (OO approach - more control than pyplot stateful)
    fig, ax = plt.subplots(figsize=(6, 4))  # figsize in inches (width, height)
    
    # Plot line connecting points
    ax.plot(df['x'], df['y'], 
            label='y vs x',          # Label for legend
            color='steelblue',       # Line color (name, hex, or RGB)
            linewidth=2)             # Line thickness
    
    # Add labels and title
    ax.set_title('Line Plot')        # Title at top
    ax.set_xlabel('x')               # X-axis label
    ax.set_ylabel('y')               # Y-axis label
    ax.legend()                      # Show legend box with labels
    
    # Adjust spacing to prevent label cutoff
    fig.tight_layout()               # Auto-adjust subplot params for clean layout
    # plt.show()                     # Uncomment to display

# --- Scatter Plot ---
def mpl_scatter_plot():
    """Scatter plot with points colored by a third variable"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create scatter plot with color mapping
    sc = ax.scatter(df['x'], df['y'],    # X, Y coordinates
                    c=df['noise'],        # Color each point by this column
                    cmap='viridis',       # Colormap (yellow-green-blue)
                    alpha=0.7)            # Transparency (0=invisible, 1=opaque)
    
    ax.set_title('Scatter Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Add colorbar to show what colors represent
    fig.colorbar(sc, ax=ax, label='noise')  # sc contains the color mapping
    fig.tight_layout()
    # plt.show()

# --- Bar Plot (categorical) ---
def mpl_bar_plot():
    """Bar chart showing frequency of categorical values"""
    # Count occurrences of each category
    counts = df['cat'].value_counts()  # Returns Series with category as index
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Create vertical bars
    ax.bar(counts.index,      # Categories on x-axis
           counts.values,     # Heights (counts) on y-axis
           color='coral')     # Bar fill color
    
    ax.set_title('Category Counts')
    ax.set_xlabel('category')
    ax.set_ylabel('count')
    fig.tight_layout()
    # plt.show()

# --- Histogram & Density ---
def mpl_hist_kde():
    """Distribution visualization with histogram and kernel density estimate"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Histogram: binned counts
    ax.hist(df['noise'], 
            bins=20,              # Number of bins (bars)
            color='slateblue', 
            alpha=0.6,            # Semi-transparent to see KDE behind
            density=True,         # Normalize to probability density (area=1)
            label='hist')
    
    # KDE: smooth density estimate (like smoothed histogram)
    sns.kdeplot(df['noise'], 
                ax=ax,            # Plot on same axes
                color='darkred', 
                label='kde')
    
    ax.set_title('Histogram + KDE')
    ax.legend()
    fig.tight_layout()
    # plt.show()

# --- Subplots grid ---
def mpl_subplots_grid():
    """Multiple plots in a grid layout - useful for comparing views"""
    # Create 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))  # 2 rows, 2 cols
    
    # Flatten 2D array to 1D for easier indexing
    axes = axes.ravel()  # [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
    
    # Plot different visualizations in each subplot
    axes[0].plot(df['x'], df['y'])
    axes[0].set_title('Line')
    
    axes[1].scatter(df['x'], df['y'])
    axes[1].set_title('Scatter')
    
    axes[2].hist(df['noise'], bins=15)
    axes[2].set_title('Hist')
    
    axes[3].bar(df['cat'].value_counts().index, df['cat'].value_counts().values)
    axes[3].set_title('Bar')
    
    # Overall title for entire figure
    fig.suptitle('Subplots Grid', fontsize=14)
    fig.tight_layout()  # Prevents overlapping labels
    # plt.show()

# --- Twin Axes (two y-scales) ---
def mpl_twin_axes():
    """Plot two variables with different scales on same x-axis"""
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    # Create second y-axis that shares the same x-axis
    ax2 = ax1.twinx()  # 'twin' the x-axis, new y-axis on right
    
    # Plot first variable on left y-axis
    ax1.plot(df['x'], df['y'], color='teal', label='y')
    
    # Plot second variable on right y-axis (different scale)
    ax2.plot(df['x'], df['noise'], color='orange', label='noise')
    
    # Label axes - color-code to match lines
    ax1.set_xlabel('x')
    ax1.set_ylabel('y', color='teal')       # Left y-axis
    ax2.set_ylabel('noise', color='orange') # Right y-axis
    
    fig.tight_layout()
    # plt.show()

# --- Save Figure ---
# Save plot to file instead of showing it
# fig.savefig('plot.png',        # Output filename
#             dpi=300,            # Resolution (dots per inch) - higher = sharper
#             bbox_inches='tight') # Crop whitespace around plot

# ============================================================================
# 2. SEABORN (HIGH-LEVEL STATISTICAL PLOTS)
# ============================================================================

# Seaborn works best with tidy DataFrames (columns = variables, rows = observations).

# --- Relational plots ---
def sns_relplot_scatter():
    """Seaborn scatter with multiple dimensions encoded via hue and size"""
    # relplot = relationship plot (scatter or line)
    sns.relplot(data=df,         # Use DataFrame (Seaborn prefers tidy data)
                x='x',           # X-axis column
                y='y',           # Y-axis column
                hue='cat',       # Color points by category
                size='noise',    # Size points by continuous variable
                palette='viridis')  # Color scheme
    plt.title('Seaborn relplot (scatter with hue/size)')
    # plt.show()

# --- Categorical plots ---
def sns_catplot_bar():
    """Bar plot showing mean values by category with error bars"""
    # catplot = categorical plot (bar, box, violin, etc.)
    sns.catplot(data=df, 
                x='cat',             # Categorical variable on x-axis
                y='y',               # Numeric variable to aggregate
                kind='bar',          # Type: bar, box, violin, swarm, etc.
                estimator=np.mean,   # How to aggregate y (mean, median, etc.)
                errorbar='sd')       # Error bars show standard deviation
    plt.title('Mean y by category with SD bars')
    # plt.show()

def sns_box_violin():
    """Compare distributions across categories using box and violin plots"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Boxplot: shows quartiles, median, outliers
    sns.boxplot(data=df, 
                x='cat',     # Categories
                y='y',       # Values to compare
                ax=axes[0])  # Plot on first subplot
    axes[0].set_title('Boxplot')
    
    # Violin plot: like boxplot but shows full distribution shape
    sns.violinplot(data=df, 
                   x='cat', 
                   y='y', 
                   ax=axes[1], 
                   inner='quartile')  # Show quartile lines inside violin
    axes[1].set_title('Violin')
    
    fig.tight_layout()
    # plt.show()

# --- Distributions ---
def sns_distplots():
    """Visualize univariate distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Histogram with KDE overlay
    sns.histplot(df['noise'],   # Data to plot
                 kde=True,       # Add kernel density estimate curve
                 bins=20,        # Number of histogram bins
                 ax=axes[0])
    axes[0].set_title('Hist + KDE')
    
    # ECDF: empirical cumulative distribution function
    # Shows what fraction of data is below each value
    sns.ecdfplot(df['noise'], ax=axes[1])
    axes[1].set_title('ECDF')
    fig.tight_layout()
    # plt.show()

# --- Joint and pair plots ---
def sns_joint_pair():
    """Explore bivariate and multivariate relationships"""
    # Joint plot: scatter + marginal distributions on edges
    sns.jointplot(data=df, 
                  x='x', 
                  y='y', 
                  kind='hex',      # Hexbin for dense data (also: 'scatter', 'reg', 'kde')
                  cmap='viridis')
    
    # Pair plot: matrix of scatter plots for all numeric pairs
    # Diagonal shows distribution of each variable
    sns.pairplot(df[['x', 'y', 'noise', 'cat']], 
                 hue='cat')  # Color by category
    # plt.show()

# --- Heatmaps (correlation) ---
def sns_heatmap_corr():
    """Visualize correlation matrix as heatmap"""
    # Calculate correlation between all numeric columns
    corr = df[['x', 'y', 'noise']].corr()  # Pearson correlation by default
    
    # Draw heatmap with correlation values
    sns.heatmap(corr, 
                annot=True,        # Show numbers in cells
                cmap='coolwarm',   # Red (positive) to blue (negative)
                fmt='.2f')         # Format numbers to 2 decimals
    plt.title('Correlation Heatmap')
    # plt.show()

# --- Faceting (small multiples) ---
def sns_facets():
    """Create grid of plots, one for each category (small multiples)"""
    # FacetGrid: create separate subplot for each unique value in 'cat'
    g = sns.FacetGrid(df, 
                      col='cat')  # One column per category (can also use row='variable')
    
    # Map a plotting function to each facet
    g.map_dataframe(sns.scatterplot, 
                    x='x', 
                    y='y', 
                    hue='cat',      # Color by category
                    legend=False)   # Legend added separately below
    
    g.add_legend()  # Add single legend for all facets
    # plt.show()

# --- Styling tips ---
# Set global Seaborn theme and defaults
# sns.set_theme(style='whitegrid',  # Background style (darkgrid, white, ticks)
#               palette='deep',      # Color palette
#               font_scale=1.1)      # Scale all fonts
# 
# Rotate axis labels for readability
# plt.xticks(rotation=45)  # Diagonal x-axis labels
# plt.yticks(rotation=0)   # Horizontal y-axis labels

# ============================================================================
# 3. PLOTLY EXPRESS (INTERACTIVE)
# ============================================================================

# Plotly Express provides concise interactive plots; works well in notebooks.

def px_scatter():
    """Interactive scatter plot with hover tooltips"""
    fig = px.scatter(df, 
                     x='x', 
                     y='y', 
                     color='cat',        # Color by category
                     size='noise',       # Point size by value
                     title='Plotly Scatter')  # Automatically interactive
    # fig.show()  # Opens in browser with zoom, pan, hover tooltips
    return fig

def px_line():
    """Interactive line plot with category coloring"""
    fig = px.line(df, 
                  x='x', 
                  y='y', 
                  color='cat',  # Separate line for each category
                  title='Plotly Line')
    # fig.show()  # Interactive legend (click to hide/show lines)
    return fig

def px_hist():
    """Interactive histogram with marginal distribution"""
    fig = px.histogram(df, 
                       x='noise', 
                       nbins=20,             # Number of bins
                       color='cat',          # Separate by category
                       marginal='box',       # Add box plot on margin (also: 'rug', 'violin')
                       title='Plotly Hist + Box')
    # fig.show()
    return fig

def px_heatmap():
    """Interactive correlation heatmap with hover values"""
    corr = df[['x', 'y', 'noise']].corr()
    fig = px.imshow(corr, 
                    text_auto='.2f',                  # Show values in cells
                    color_continuous_scale='RdBu_r',  # Red-blue diverging (reversed)
                    title='Correlation Heatmap')
    # fig.show()  # Hover shows exact correlation values
    return fig

# ============================================================================
# 4. GENERAL PATTERNS & TIPS
# ============================================================================

# DATA PREPARATION:
# - Keep data tidy: columns = variables, rows = observations
# - Clean missing values before plotting (dropna or fillna)

# DESIGN PRINCIPLES:
# - Always label axes and add titles
# - Use legends wisely (avoid clutter, place strategically)
# - Annotate key points if needed (max/min, inflection points)
# - Use color palettes consistently; be mindful of colorblind-safe palettes
#   (e.g., viridis, colorbrewer palettes)

# CLARITY & READABILITY:
# - Prefer faceting over overplotting: small multiples show subgroups clearly
# - For large scatter plots, use alpha<1 and smaller markers; consider hexbin/kde
# - Use tight_layout() or constrained_layout=True to reduce label overlap
# - Rotate axis labels when needed (dates, long category names)

# TECHNICAL TIPS:
# - Save with high dpi for print: fig.savefig('plot.png', dpi=300, bbox_inches='tight')
# - Interactive needs: Plotly or mplcursors (Matplotlib) for hover info
# - Reuse style: sns.set_theme() or plt.style.use('seaborn-v0_8') for consistent appearance
# - For animations or real-time updates, use FuncAnimation (matplotlib.animation)

# ============================================================================
# 5. QUICK RECIPE COLLECTION
# ============================================================================

# --- Annotate points ---
def annotate_example():
    """Add text labels to specific data points"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df['x'], df['y'], alpha=0.6)
    
    # Find and annotate maximum y value
    idx = df['y'].idxmax()  # Index of max y
    ax.annotate('max y',                           # Label text
                xy=(df.loc[idx, 'x'], df.loc[idx, 'y']),  # Point to annotate
                xytext=(10, 10),                   # Offset from point (in points)
                textcoords='offset points',        # How to interpret xytext
                arrowprops=dict(arrowstyle='->', color='black'))  # Arrow style
    fig.tight_layout()
    # plt.show()

# --- Horizontal / vertical reference lines ---
def ref_lines():
    """Add reference lines to show thresholds or summary statistics"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['x'], df['y'])
    
    # Horizontal line at mean y value
    ax.axhline(df['y'].mean(),        # y-coordinate for horizontal line
               color='red', 
               linestyle='--',        # Dashed line
               label='mean y')
    
    # Vertical line at median x value
    ax.axvline(df['x'].median(),      # x-coordinate for vertical line
               color='green', 
               linestyle=':',         # Dotted line
               label='median x')
    
    ax.legend()
    fig.tight_layout()
    # plt.show()

# --- Customizing ticks ---
def ticks_custom():
    """Control exact tick positions and formatting"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['x'], df['y'])
    
    # Set specific tick positions
    ax.set_xticks(np.arange(0, 11, 2))  # X ticks at 0, 2, 4, 6, 8, 10
    ax.set_yticks(np.arange(int(df['y'].min()), int(df['y'].max()) + 1, 5))  # Every 5 units
    
    # Rotate tick labels
    ax.tick_params(axis='x', rotation=0)  # 0=horizontal, 45=diagonal, 90=vertical
    
    fig.tight_layout()
    # plt.show()

# --- Log scales ---
def log_scale():
    """Use logarithmic scale when data spans orders of magnitude"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot exponential/power-law data
    ax.plot(df['x'], np.abs(df['noise']) + 1)  # +1 to avoid log(0)
    
    # Switch y-axis to log scale (makes exponential curves linear)
    ax.set_yscale('log')  # Also available: 'linear', 'symlog', 'logit'
    ax.set_title('Log Y Scale')
    
    fig.tight_layout()
    # plt.show()

# --- Multiple figures saving loop example ---
def save_multiple():
    """Generate and save separate plots for each group"""
    # Loop through each category
    for cat_val, sub in df.groupby('cat'):
        # Create individual plot for this category
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(sub['x'], sub['y'], alpha=0.7)
        ax.set_title(f'Scatter for {cat_val}')
        fig.tight_layout()
        
        # Save to file with category in filename
        # fig.savefig(f'scatter_{cat_val}.png', dpi=200)
        
        plt.close(fig)  # Close to free memory (important in loops!)

# ============================================================================
# 6. TIME SERIES VISUALIZATION
# ============================================================================

# Time series data often needs special date handling and rolling/resampling plots.

def timeseries_basic():
    """Date formatting, resampling, rolling windows for time series"""
    import matplotlib.dates as mdates  # For date formatting on axes
    
    # Create sample time series data
    dates = pd.date_range('2023-01-01', periods=365, freq='D')  # Daily dates
    ts_data = pd.DataFrame({
        'date': dates,
        'value': np.cumsum(np.random.randn(365)) + 100,  # Random walk
        'category': np.random.choice(['A', 'B'], 365)
    })
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Subplot 1: Basic time series line plot
    axes[0].plot(ts_data['date'], ts_data['value'], linewidth=1)
    axes[0].set_title('Time Series - Daily Values')
    axes[0].set_ylabel('Value')
    
    # Subplot 2: Rolling mean (smooths out noise)
    # Calculate 30-day moving average
    rolling = ts_data.set_index('date')['value'].rolling(window=30).mean()
    axes[1].plot(ts_data['date'], ts_data['value'], alpha=0.3, label='Daily')
    axes[1].plot(rolling.index,   # Rolling returns indexed series
                 rolling.values, 
                 color='red', 
                 linewidth=2, 
                 label='30-day MA')  # Moving average
    axes[1].set_title('Time Series with Rolling Mean')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    
    # Subplot 3: Monthly resampling (aggregation)
    # Downsample daily to monthly by taking mean
    monthly = ts_data.set_index('date')['value'].resample('M').mean()  # 'M'=month end
    axes[2].bar(monthly.index, 
                monthly.values, 
                width=20,       # Bar width in days
                alpha=0.7)
    axes[2].set_title('Monthly Average Values')
    axes[2].set_ylabel('Value')
    
    # Format x-axis dates for all subplots
    for ax in axes:
        # Show dates as 'YYYY-MM' format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        # Place ticks every 2 months
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        # Rotate labels for readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    fig.tight_layout()
    # plt.show()

def timeseries_area_stacked():
    """Stacked area chart shows composition over time (parts of a whole)"""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = {
        'date': dates,
        'A': np.random.rand(100).cumsum(),  # Cumulative sum creates trending data
        'B': np.random.rand(100).cumsum(),
        'C': np.random.rand(100).cumsum()
    }
    df_ts = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Stack areas on top of each other (bottom to top: A, B, C)
    ax.stackplot(df_ts['date'],      # X-axis (time)
                 df_ts['A'],         # First series (bottom)
                 df_ts['B'],         # Second series (middle)
                 df_ts['C'],         # Third series (top)
                 labels=['A', 'B', 'C'], 
                 alpha=0.7)          # Transparency
    
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
    """Fit regression line with confidence interval for predictions"""
    # Seaborn makes regression plotting easy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Linear regression with 95% confidence interval (shaded region)
    sns.regplot(data=df, 
                x='x', 
                y='y', 
                ax=axes[0], 
                scatter_kws={'alpha': 0.5})  # Make points semi-transparent
    axes[0].set_title('Regression with 95% CI (Seaborn)')
    
    # Polynomial regression (order=2 means quadratic: y = ax² + bx + c)
    sns.regplot(data=df, 
                x='x', 
                y='y', 
                ax=axes[1], 
                order=2,                     # Polynomial degree
                scatter_kws={'alpha': 0.5})
    axes[1].set_title('Polynomial Regression (order=2)')
    
    fig.tight_layout()
    # plt.show()

def statistical_residuals():
    """Diagnostic plots for checking regression assumptions"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Residual plot: check if residuals are randomly scattered
    # (should have no pattern if model is appropriate)
    sns.residplot(data=df, 
                  x='x', 
                  y='y', 
                  ax=axes[0], 
                  lowess=True,      # Add smoothed trend line
                  color='steelblue')
    axes[0].set_title('Residual Plot')
    axes[0].axhline(0,              # Reference line at zero
                    color='red', 
                    linestyle='--', 
                    linewidth=1)
    
    # Q-Q plot: check if data follows normal distribution
    # Points should fall on diagonal line if normally distributed
    from scipy import stats
    stats.probplot(df['noise'],     # Data to test
                   dist="norm",     # Compare to normal distribution
                   plot=axes[1])    # Plot on axes[1]
    axes[1].set_title('Q-Q Plot')
    
    fig.tight_layout()
    # plt.show()

def statistical_error_bars():
    """Visualize uncertainty with error bars"""
    # Calculate statistics for each category
    grouped = df.groupby('cat')['y'].agg([
        'mean',  # Average
        'std',   # Standard deviation (spread)
        'sem'    # Standard error of mean (uncertainty in mean)
    ])
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot 1: Error bars showing standard deviation
    axes[0].errorbar(grouped.index,      # Categories on x-axis
                     grouped['mean'],    # Mean values
                     yerr=grouped['std'], # Error bars = ±1 std dev
                     fmt='o',            # Circle markers
                     capsize=5,          # Width of error bar caps
                     capthick=2,         # Thickness of caps
                     markersize=8)
    axes[0].set_title('Mean ± Std Dev')
    axes[0].set_ylabel('Value')
    
    # Plot 2: Error bars showing standard error (smaller, more precise)
    axes[1].errorbar(grouped.index, 
                     grouped['mean'], 
                     yerr=grouped['sem'],  # SEM typically smaller than std
                     fmt='s',              # Square markers
                     capsize=5, 
                     capthick=2, 
                     markersize=8, 
                     color='coral')
    axes[1].set_title('Mean ± SEM')
    axes[1].set_ylabel('Value')
    
    # Plot 3: Bar chart with error bars (common in publications)
    axes[2].bar(grouped.index,       # Categories
                grouped['mean'],     # Bar heights
                yerr=grouped['std'], # Error bars
                capsize=5,           # Error bar cap width
                alpha=0.7,           # Bar transparency
                color='steelblue')
    axes[2].set_title('Bar Chart with Error Bars')
    axes[2].set_ylabel('Value')
    
    fig.tight_layout()
    # plt.show()

def statistical_swarm_strip():
    """Show all individual points for categorical data"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Strip plot: randomly jittered points to avoid overlap
    sns.stripplot(data=df, 
                  x='cat',       # Categories
                  y='y',         # Values
                  ax=axes[0], 
                  alpha=0.6, 
                  jitter=True)   # Add random x-displacement
    axes[0].set_title('Strip Plot (Jittered Points)')
    
    # Swarm plot: deterministically arranged to avoid overlap
    # Better for showing distribution, but slower for large datasets
    sns.swarmplot(data=df, 
                  x='cat', 
                  y='y', 
                  ax=axes[1], 
                  alpha=0.7)
    axes[1].set_title('Swarm Plot (Non-overlapping)')
    
    fig.tight_layout()
    # plt.show()

# ============================================================================
# 8. CONTOUR AND 3D PLOTS
# ============================================================================

def contour_plots():
    """Contour plots show 3D surface as 2D with level curves"""
    # Create 2D grid of x,y coordinates
    x_grid = np.linspace(-3, 3, 100)  # 100 points from -3 to 3
    y_grid = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_grid, y_grid)  # Create 2D arrays of coordinates
    
    # Calculate z values for each (x,y) point
    Z = np.sin(np.sqrt(X**2 + Y**2))  # Ripple function
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot 1: Contour lines only (like topographic map)
    cs = axes[0].contour(X, Y, Z,      # Grid and values
                         levels=10,     # Number of contour lines
                         cmap='viridis')
    axes[0].clabel(cs,                 # Add labels to contour lines
                   inline=True,        # Label on the line itself
                   fontsize=8)
    axes[0].set_title('Contour Lines')
    
    # Plot 2: Filled contours (regions between levels colored)
    cf = axes[1].contourf(X, Y, Z,     # 'f' = filled
                          levels=20,    # More levels = smoother
                          cmap='viridis')
    fig.colorbar(cf, ax=axes[1])       # Show color scale
    axes[1].set_title('Filled Contour')
    
    # Plot 3: Contour with data points overlaid
    axes[2].contourf(X, Y, Z, 
                     levels=20, 
                     cmap='viridis', 
                     alpha=0.6)        # Semi-transparent background
    # Overlay scatter points
    axes[2].scatter(df['x'][:50] - 5,  # Shift x to fit in range
                    df['noise'][:50], 
                    c='red', 
                    s=20, 
                    alpha=0.8)
    axes[2].set_title('Contour + Scatter')
    
    fig.tight_layout()
    # plt.show()

def plot_3d():
    """Create 3D visualizations for multivariate data"""
    from mpl_toolkits.mplot3d import Axes3D  # Import 3D projection
    
    fig = plt.figure(figsize=(14, 5))
    
    # Plot 1: 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')  # 1 row, 3 cols, position 1
    ax1.scatter(df['x'],       # X coordinates
                df['y'],       # Y coordinates
                df['noise'],   # Z coordinates
                c=df['noise'], # Color by z-value
                cmap='plasma', 
                alpha=0.6)
    ax1.set_title('3D Scatter')
    # Label all three axes
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Noise')
    
    # Plot 2: 3D surface plot (continuous surface)
    ax2 = fig.add_subplot(132, projection='3d')
    x_grid = np.linspace(-3, 3, 50)
    y_grid = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_grid, y_grid)  # Create 2D grid
    Z = np.sin(np.sqrt(X**2 + Y**2))    # Calculate z for each (x,y)
    
    # Draw surface with color mapping
    surf = ax2.plot_surface(X, Y, Z, 
                            cmap='coolwarm',  # Color scheme
                            alpha=0.8)        # Transparency
    ax2.set_title('3D Surface')
    fig.colorbar(surf, ax=ax2, shrink=0.5)  # shrink=scale colorbar size
    
    # Plot 3: 3D wireframe (mesh/grid view)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_wireframe(X, Y, Z, 
                       color='steelblue', 
                       alpha=0.5)  # See-through mesh
    ax3.set_title('3D Wireframe')
    
    fig.tight_layout()
    # plt.show()

# ============================================================================
# 9. PIE CHARTS & COMPOSITION
# ============================================================================

def pie_donut_charts():
    """Pie and donut charts for composition (use sparingly - bars often clearer)"""
    counts = df['cat'].value_counts()  # Get category frequencies
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Basic pie chart: circular sectors proportional to values
    axes[0].pie(counts.values,              # Sizes of slices
                labels=counts.index,        # Labels for each slice
                autopct='%1.1f%%',          # Show percentages (1 decimal)
                startangle=90,              # Rotate start position
                colors=sns.color_palette('pastel'))  # Soft colors
    axes[0].set_title('Pie Chart')
    
    # Donut chart: pie with center cut out (emphasizes arc length vs area)
    axes[1].pie(counts.values, 
                labels=counts.index, 
                autopct='%1.1f%%',
                startangle=90, 
                colors=sns.color_palette('pastel'),
                wedgeprops={'width': 0.4})  # width < 1 creates donut hole
    axes[1].set_title('Donut Chart')
    
    fig.tight_layout()
    # plt.show()

# ============================================================================
# 10. GEOGRAPHIC MAPS
# ============================================================================

def geographic_maps():
    """Create choropleth and scatter maps for geographic data"""
    # Sample state-level data (US states)
    state_data = pd.DataFrame({
        'state': ['CA', 'TX', 'FL', 'NY', 'PA'],  # State abbreviations
        'value': [100, 85, 70, 90, 60]            # Data to visualize
    })
    
    # Choropleth map: regions colored by value
    fig1 = px.choropleth(
        state_data,
        locations='state',         # Column with location identifiers
        locationmode='USA-states', # How to interpret locations
        color='value',             # Column to map to colors
        scope='usa',               # Map extent (usa, world, europe, etc.)
        title='Choropleth Map - US States'
    )
    # fig1.show()  # Opens interactive map in browser
    
    # Scatter map: points at geographic coordinates
    city_data = pd.DataFrame({
        'city': ['New York', 'Los Angeles', 'Chicago', 'Houston'],
        'lat': [40.7128, 34.0522, 41.8781, 29.7604],   # Latitudes
        'lon': [-74.0060, -118.2437, -87.6298, -95.3698],  # Longitudes
        'population': [8.4, 3.9, 2.7, 2.3]  # Millions, mapped to size
    })
    
    fig2 = px.scatter_geo(
        city_data,
        lat='lat',              # Latitude column
        lon='lon',              # Longitude column
        size='population',      # Point size by value
        hover_name='city',      # Show city name on hover
        scope='usa',
        title='Scatter Map - US Cities'
    )
    # fig2.show()
    
    return fig1, fig2

# ============================================================================
# 11. ADVANCED LAYOUTS & CUSTOMIZATION
# ============================================================================

def advanced_subplots_gridspec():
    """GridSpec allows flexible, non-uniform subplot layouts"""
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(10, 8))
    
    # Create 3x3 grid with custom spacing
    gs = GridSpec(3, 3,                # 3 rows, 3 columns
                  figure=fig, 
                  hspace=0.4,          # Vertical spacing between plots
                  wspace=0.4)          # Horizontal spacing
    
    # Large subplot spanning rows 0-1, columns 0-1 (2x2 grid)
    ax1 = fig.add_subplot(gs[0:2, 0:2])  # Slice notation for span
    ax1.plot(df['x'], df['y'], 'o-', alpha=0.6)
    ax1.set_title('Main Plot (2x2)')
    
    # Small subplot: row 0, column 2
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(df['noise'], bins=15, color='coral', alpha=0.7)
    ax2.set_title('Hist')
    
    # Small subplot: row 1, column 2
    ax3 = fig.add_subplot(gs[1, 2])
    # Create boxplot from grouped data
    ax3.boxplot([df[df['cat'] == c]['y'].values for c in ['A', 'B', 'C']])
    ax3.set_title('Box')
    
    # Wide subplot spanning row 2, all columns
    ax4 = fig.add_subplot(gs[2, :])  # : means all columns
    ax4.scatter(df['x'], df['y'], c=df['noise'], cmap='viridis', alpha=0.6)
    ax4.set_title('Bottom Span (1x3)')
    
    # plt.show()

def inset_axes_example():
    """Create small plot inside main plot (picture-in-picture)"""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['x'], df['y'], 'o-', alpha=0.6)
    ax.set_title('Main Plot with Inset')
    
    # Create inset axes within main axes
    ax_inset = inset_axes(ax,                # Parent axes
                          width="40%",       # Width as % of parent
                          height="30%",      # Height as % of parent
                          loc='upper right') # Position (1-10, or string)
    
    # Plot in the inset
    ax_inset.hist(df['noise'], bins=15, color='steelblue', alpha=0.7)
    ax_inset.set_title('Inset Hist', fontsize=10)
    
    fig.tight_layout()
    # plt.show()

def custom_legends_colorbars():
    """Fine-tune legend placement and colorbar formatting"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Legend positioned outside plot area
    for cat_val in df['cat'].unique():
        subset = df[df['cat'] == cat_val]
        axes[0].scatter(subset['x'], subset['y'], 
                       label=f'Cat {cat_val}', 
                       alpha=0.7, 
                       s=50)
    
    # Place legend outside right edge of plot
    axes[0].legend(loc='upper left',       # Anchor point on legend box
                   bbox_to_anchor=(1.02, 1),  # Position relative to axes (x, y)
                   borderaxespad=0)       # Padding between axes and legend
    axes[0].set_title('Legend Outside')
    
    # Plot 2: Customized colorbar
    sc = axes[1].scatter(df['x'], df['y'], 
                         c=df['noise'],    # Color by this variable
                         cmap='coolwarm',  # Diverging colormap
                         s=50)
    
    cbar = fig.colorbar(sc, ax=axes[1])   # Add colorbar
    cbar.set_label('Noise Level',         # Colorbar label
                   rotation=270,           # Rotate label vertical
                   labelpad=20)            # Distance from colorbar
    axes[1].set_title('Custom Colorbar')
    
    fig.tight_layout()
    # plt.show()

# ============================================================================
# 12. ANIMATION BASICS
# ============================================================================

def animation_simple():
    """Create animated plot that updates over time"""
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=(6, 4))
    xdata, ydata = [], []  # Lists to accumulate data points
    
    # Create line object (initially empty)
    ln, = ax.plot([], [], 'o-', animated=True)  # Comma unpacks single-element list
    
    def init():
        """Initialize animation: set plot limits"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 30)
        return ln,  # Return tuple of artists to update
    
    def update(frame):
        """Update function called for each frame"""
        # Add new data point
        xdata.append(frame)
        ydata.append(frame**0.5 * 5 + np.random.randn())  # Some function with noise
        
        # Update line data
        ln.set_data(xdata, ydata)
        return ln,  # Return modified artists
    
    # Create animation
    ani = FuncAnimation(fig,              # Figure to animate
                       update,            # Update function
                       frames=np.linspace(0, 10, 50),  # Frame values (passed to update)
                       init_func=init,    # Initialization function
                       blit=True,         # Optimize by only redrawing changed parts
                       interval=50)       # Delay between frames (milliseconds)
    
    # Save animation to file
    # ani.save('animation.gif', writer='pillow', fps=20)
    # ani.save('animation.mp4', writer='ffmpeg', fps=20)
    
    # plt.show()  # Display animation
    return ani  # Return to prevent garbage collection

# ============================================================================
# 13. SEABORN ADVANCED PATTERNS
# ============================================================================

def seaborn_advanced():
    """Advanced Seaborn: count plots, point plots, custom estimators, overlays"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Count plot (bar chart of frequencies)
    sns.countplot(data=df, 
                  x='cat',           # Count occurrences of each category
                  ax=axes[0, 0], 
                  palette='Set2')    # Color palette
    axes[0, 0].set_title('Count Plot')
    
    # Plot 2: Point plot (shows means connected by lines)
    sns.pointplot(data=df, 
                  x='cat', 
                  y='y', 
                  ax=axes[0, 1], 
                  errorbar='sd',     # Error bars = standard deviation
                  capsize=0.1)       # Size of error bar caps
    axes[0, 1].set_title('Point Plot with SD')
    
    # Plot 3: Bar plot with custom aggregation function
    sns.barplot(data=df, 
                x='cat', 
                y='y', 
                estimator=np.median,  # Use median instead of mean
                errorbar=('pi', 75),  # 75% prediction interval (not CI)
                ax=axes[1, 0], 
                palette='muted')
    axes[1, 0].set_title('Bar Plot - Median with 75% PI')
    
    # Plot 4: Overlay box plot with individual points (strip plot)
    # Box plot first (background)
    sns.boxplot(data=df, 
                x='cat', 
                y='y', 
                ax=axes[1, 1], 
                width=0.3,         # Narrow boxes to make room for points
                palette='pastel')
    
    # Strip plot on top (overlay)
    sns.stripplot(data=df, 
                  x='cat', 
                  y='y', 
                  ax=axes[1, 1], 
                  color='black',   # All points same color
                  alpha=0.3,       # Transparent to see overlap
                  size=3)          # Small points
    axes[1, 1].set_title('Box + Strip Overlay')
    
    fig.tight_layout()
    # plt.show()

# ============================================================================
# 14. COLOR & STYLE NOTES
# ============================================================================

# COLORMAPS:
# - Perceptually uniform (good default): viridis, plasma, magma, inferno, cividis
# - Diverging (for data centered around zero): coolwarm, RdBu_r, seismic
# - Sequential (low to high): Blues, Greens, YlOrRd
# - Qualitative (distinct categories): tab10, Set1, Set2, Pastel1, Accent
# - Access via: plt.cm.viridis or cmap='viridis' in plot functions

# SEABORN COLOR PALETTES:
# - Default palettes: 'deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'
# - Usage: sns.color_palette('deep') or palette='deep' in Seaborn functions
# - Diverging: sns.diverging_palette() for custom ranges
# - Sequential: sns.light_palette(), sns.dark_palette()

# COLORBLIND-FRIENDLY:
# - Use colorblind palette: sns.color_palette('colorblind')
# - Avoid red-green combinations alone
# - Add patterns/markers for redundancy

# STYLE CONTEXT:
# - Set global theme: sns.set_theme(style='darkgrid', palette='deep', font_scale=1.2)
# - Matplotlib styles: plt.style.use('seaborn-v0_8'), 'ggplot', 'fivethirtyeight'
# - Temporary context: with sns.axes_style('whitegrid'): ...

# ============================================================================
# 15. SPECIALIZED VISUALIZATIONS
# ============================================================================
# Advanced plot types for specific use cases: hierarchical data, multivariate
# comparisons, flow diagrams, and network relationships.

def dendrogram_clustering():
    """Dendrogram for hierarchical clustering - shows how data points group"""
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    # Create sample data for clustering (2D points)
    np.random.seed(42)
    # Generate 3 clusters of points
    cluster1 = np.random.randn(10, 2) + np.array([0, 0])
    cluster2 = np.random.randn(10, 2) + np.array([5, 5])
    cluster3 = np.random.randn(10, 2) + np.array([0, 5])
    data = np.vstack([cluster1, cluster2, cluster3])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: The original data points
    axes[0].scatter(data[:, 0], data[:, 1], alpha=0.6, s=100)
    axes[0].set_title('Original Data Points')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # Plot 2: Hierarchical clustering dendrogram
    # Compute linkage matrix (how to merge clusters)
    linkage_matrix = linkage(data, method='ward')  # Ward minimizes variance
    
    # Create dendrogram
    dendrogram(linkage_matrix,      # Linkage matrix from above
               ax=axes[1],          # Plot on second subplot
               orientation='top',   # Tree grows downward
               distance_sort='ascending',  # Sort by distance
               show_leaf_counts=True)      # Show number of items in each leaf
    
    axes[1].set_title('Hierarchical Clustering Dendrogram')
    axes[1].set_xlabel('Data Point Index (or Cluster)')
    axes[1].set_ylabel('Distance (Height)')
    
    # Note: Height shows distance where clusters merge
    # Can "cut" tree at any height to get desired number of clusters
    
    fig.tight_layout()
    # plt.show()

def radar_chart():
    """Radar/Spider chart for multivariate comparison on same scale"""
    # Example: Compare products/items across multiple attributes
    
    # Sample data: 3 products rated on 5 attributes (scale 0-10)
    categories = ['Quality', 'Price', 'Support', 'Features', 'Speed']
    n_cats = len(categories)
    
    # Ratings for each product
    product_a = [8, 6, 7, 9, 8]  # Product A scores
    product_b = [6, 9, 5, 7, 6]  # Product B scores
    product_c = [7, 7, 9, 6, 7]  # Product C scores
    
    # Compute angle for each axis (evenly spaced around circle)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    
    # Close the plot by appending first value to end
    product_a += [product_a[0]]
    product_b += [product_b[0]]
    product_c += [product_c[0]]
    angles += [angles[0]]
    
    # Create polar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot each product as a line on radar chart
    ax.plot(angles, product_a, 'o-', linewidth=2, label='Product A', color='blue')
    ax.fill(angles, product_a, alpha=0.15, color='blue')  # Fill area
    
    ax.plot(angles, product_b, 's-', linewidth=2, label='Product B', color='red')
    ax.fill(angles, product_b, alpha=0.15, color='red')
    
    ax.plot(angles, product_c, '^-', linewidth=2, label='Product C', color='green')
    ax.fill(angles, product_c, alpha=0.15, color='green')
    
    # Set category labels at each spoke
    ax.set_xticks(angles[:-1])  # Exclude duplicate first angle
    ax.set_xticklabels(categories)
    
    # Set radial limits and labels
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], color='gray', size=8)
    
    ax.set_title('Product Comparison Radar Chart', size=16, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    fig.tight_layout()
    # plt.show()

def waterfall_chart():
    """Waterfall chart shows cumulative effect of sequential changes"""
    # Use case: financial analysis, showing how components add up to total
    
    # Example: Revenue breakdown
    categories = ['Starting\nRevenue', 'Product\nSales', 'Services', 
                  'Costs', 'Taxes', 'Final\nProfit']
    values = [100, 50, 30, -60, -20, 0]  # Changes (positive/negative)
    
    # Calculate cumulative values and bar positions
    cumulative = [0]  # Starting position
    for val in values[:-1]:  # All except final (which we'll calculate)
        cumulative.append(cumulative[-1] + val)
    
    # Final value is last cumulative
    values[-1] = cumulative[-1]  # Final profit = cumulative total
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars with different colors for increase/decrease
    colors = ['blue', 'green', 'green', 'red', 'red', 'blue']
    
    # For each bar, plot from previous cumulative to current
    x_pos = np.arange(len(categories))
    
    for i, (cat, val, cum) in enumerate(zip(categories, values, cumulative)):
        if i == 0:  # Starting value - full bar from 0
            ax.bar(i, val, color=colors[i], alpha=0.7, width=0.6)
        elif i == len(categories) - 1:  # Final value - full bar from 0
            ax.bar(i, val, color=colors[i], alpha=0.7, width=0.6)
        else:  # Intermediate - floating bar showing change
            bar_bottom = cum  # Start from previous cumulative
            ax.bar(i, val, bottom=bar_bottom, color=colors[i], alpha=0.7, width=0.6)
            # Draw connector line to next bar
            if i < len(categories) - 1:
                ax.plot([i + 0.3, i + 0.7], [cum + val, cum + val], 
                       'k--', linewidth=1, alpha=0.5)
        
        # Add value labels on bars
        if i < len(categories) - 1:
            label_y = cum + val / 2 if val > 0 else cum + val / 2
            ax.text(i, label_y, f'{val:+.0f}', ha='center', va='center', 
                   fontweight='bold', fontsize=10)
        else:
            ax.text(i, val / 2, f'{val:.0f}', ha='center', va='center', 
                   fontweight='bold', fontsize=10)
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Value ($M)', fontsize=12)
    ax.set_title('Waterfall Chart - Revenue to Profit Breakdown', fontsize=14)
    ax.axhline(0, color='black', linewidth=0.8)  # Zero line
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    # plt.show()

def sankey_diagram():
    """Sankey diagram shows flow quantities between nodes (requires plotly)"""
    import plotly.graph_objects as go  # Lower-level Plotly for Sankey
    
    # Example: Energy flow from sources to uses
    # Define nodes (sources and targets)
    labels = ['Coal', 'Natural Gas', 'Nuclear', 'Renewable',  # Sources (0-3)
              'Electricity', 'Heat',                            # Intermediate (4-5)
              'Residential', 'Commercial', 'Industrial']        # End uses (6-8)
    
    # Define flows: source index -> target index with value
    sources = [0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5]  # From node indices
    targets = [4, 5, 4, 5, 4, 4, 5, 6, 7, 8, 6, 7, 8]  # To node indices
    values =  [40, 10, 30, 15, 25, 20, 5, 30, 25, 60, 10, 10, 10]  # Flow amounts
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        # Node styling
        node=dict(
            pad=15,           # Spacing between nodes
            thickness=20,     # Node width
            line=dict(color='black', width=0.5),
            label=labels,     # Node labels
            color='lightblue' # Node colors
        ),
        # Link (flow) styling
        link=dict(
            source=sources,   # Start nodes (by index)
            target=targets,   # End nodes (by index)
            value=values,     # Flow magnitude (width of ribbon)
            color='rgba(0, 150, 255, 0.3)'  # Semi-transparent flow color
        )
    )])
    
    fig.update_layout(
        title='Energy Flow Sankey Diagram',
        font=dict(size=12),
        height=600
    )
    
    # fig.show()  # Opens in browser
    return fig

def network_graph():
    """Network/graph visualization showing nodes and connections (requires networkx)"""
    try:
        import networkx as nx  # Graph/network library
    except ImportError:
        print("NetworkX not installed. Install with: pip install networkx")
        return
    
    # Create sample network (social network, citation network, etc.)
    G = nx.Graph()  # Undirected graph
    
    # Add nodes (people, papers, websites, etc.)
    nodes = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank', 'Grace']
    G.add_nodes_from(nodes)
    
    # Add edges (connections/relationships)
    edges = [
        ('Alice', 'Bob'), ('Alice', 'Carol'), ('Alice', 'Dave'),
        ('Bob', 'Carol'), ('Bob', 'Eve'),
        ('Carol', 'Dave'), ('Carol', 'Frank'),
        ('Dave', 'Frank'),
        ('Eve', 'Frank'), ('Eve', 'Grace'),
        ('Frank', 'Grace')
    ]
    G.add_edges_from(edges)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Layout 1: Spring layout (force-directed, nodes repel/attract)
    pos1 = nx.spring_layout(G, seed=42)  # Seed for reproducibility
    
    # Draw network on first subplot
    nx.draw(G, pos1, ax=axes[0],
            with_labels=True,          # Show node names
            node_color='lightblue',    # Node fill color
            node_size=1000,            # Node size
            font_size=10,              # Label font size
            font_weight='bold',        # Label font weight
            edge_color='gray',         # Connection color
            width=2,                   # Edge thickness
            alpha=0.8)                 # Transparency
    axes[0].set_title('Network Graph - Spring Layout')
    
    # Layout 2: Circular layout
    pos2 = nx.circular_layout(G)
    
    # Calculate node sizes based on degree (number of connections)
    degrees = dict(G.degree())  # Number of connections per node
    node_sizes = [degrees[node] * 300 for node in G.nodes()]  # Scale by degree
    
    nx.draw(G, pos2, ax=axes[1],
            with_labels=True,
            node_color=list(degrees.values()),  # Color by degree
            node_size=node_sizes,               # Size by degree
            cmap='viridis',                     # Color map
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            width=2,
            alpha=0.8)
    axes[1].set_title('Network Graph - Circular Layout (sized by connections)')
    
    fig.tight_layout()
    # plt.show()
    
    # Print network statistics
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Average degree: {sum(degrees.values()) / len(degrees):.2f}")

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
    print(" 14. Color & style notes")
    print(" 15. Specialized visualizations (dendrograms, radar, waterfall, Sankey, networks)")
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
    
    # Advanced sections
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
    
    # Specialized visualizations
    dendrogram_clustering()
    radar_chart()
    waterfall_chart()
    sankey_diagram()
    network_graph()
    
    print("\nAll visualization examples loaded successfully!")
    print("Explore each function to learn specific patterns.")
    print("\nNote: Some specialized plots require additional packages:")
    print("  - NetworkX: pip install networkx")
    print("  - Plotly for Sankey: pip install plotly (already imported)")
