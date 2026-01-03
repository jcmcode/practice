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
# 6. COLOR & STYLE NOTES
# ============================================================================

# Matplotlib built-ins: plt.cm.viridis, plasma, magma, cividis, coolwarm, etc.
# Seaborn palettes: sns.color_palette('deep'), 'muted', 'bright', 'dark', 'colorblind'
# Diverging palettes for centered data: 'RdBu_r', 'coolwarm'
# Qualitative for categories: 'tab10', 'Set2', 'Pastel1'

# ============================================================================
# MAIN DEMO (lightweight sanity checks)
# ============================================================================

if __name__ == "__main__":
    print("Visualization cheat sheet loaded. Uncomment plt.show() lines to view plots.")
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
