import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# -------------------------------
# Simulate time series with causal relationship
# -------------------------------
np.random.seed(42)
time = pd.date_range(start='2025-01-01', periods=200, freq='H')

# Series A: base signal
series_a = np.sin(np.linspace(0, 10 * np.pi, 200)) + np.random.normal(0, 0.3, 200)

# Series B: influenced by series A + noise
series_b = 0.8 * series_a + np.random.normal(0, 0.2, 200)

df = pd.DataFrame({'Time': time, 'SeriesA': series_a, 'SeriesB': series_b})

# Rolling correlation
window_size = 20
df['RollingCorr'] = df['SeriesA'].rolling(window=window_size).corr(df['SeriesB'])

# -------------------------------
# Main line plot figure
# -------------------------------
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.set_xlim(0, len(df))
ax1.set_ylim(df[['SeriesA', 'SeriesB']].min().min() - 0.5, df[['SeriesA', 'SeriesB']].max().max() + 0.5)
ax1.set_ylabel('Value')
ax1.set_title('Series A vs Series B')

line_a, = ax1.plot([], [], color='blue', lw=2, label='Series A')
line_b, = ax1.plot([], [], color='orange', lw=2, label='Series B')
ax1.legend(loc='upper left')

ax2.set_xlim(0, len(df))
ax2.set_ylim(-1, 1)
ax2.set_ylabel('Rolling Corr')
ax2.set_xlabel('Time')
ax2.set_title('Rolling Correlation (Window=20)')
corr_line, = ax2.plot([], [], color='green', lw=2)

# -------------------------------
# Heatmap figure
# -------------------------------
fig2, ax_heat = plt.subplots(figsize=(5, 5))
heatmap_data = np.zeros((2, 2))
im = ax_heat.imshow(heatmap_data, vmin=-1, vmax=1, cmap='RdYlGn')
ax_heat.set_xticks([0, 1])
ax_heat.set_yticks([0, 1])
ax_heat.set_xticklabels(['SeriesA', 'SeriesB'])
ax_heat.set_yticklabels(['SeriesA', 'SeriesB'])
ax_heat.set_title('Rolling Correlation Heatmap')

# -------------------------------
# Animation function
# -------------------------------
x_data, y_a, y_b, y_corr = [], [], [], []

def update(frame):
    # Update line plots
    x_data.append(frame)
    y_a.append(df['SeriesA'].iloc[frame])
    y_b.append(df['SeriesB'].iloc[frame])
    y_corr.append(df['RollingCorr'].iloc[frame] if not np.isnan(df['RollingCorr'].iloc[frame]) else 0)

    line_a.set_data(x_data, y_a)
    line_b.set_data(x_data, y_b)
    corr_line.set_data(x_data, y_corr)

    # Update heatmap (2x2 correlation matrix)
    if frame >= window_size:
        window_df = df[['SeriesA', 'SeriesB']].iloc[frame-window_size:frame]
        corr_matrix = window_df.corr().values
    else:
        corr_matrix = np.zeros((2, 2))
    im.set_data(corr_matrix)

    return line_a, line_b, corr_line, im

# -------------------------------
# Create animation
# -------------------------------
ani = FuncAnimation(
    fig1, update, frames=len(df), blit=True, repeat=False
)

ani_heat = FuncAnimation(
    fig2, update, frames=len(df), blit=True, repeat=False
)

# Save GIFs
ani.save("time_series_correlation.gif", writer=PillowWriter(fps=10))
ani_heat.save("rolling_corr_heatmap.gif", writer=PillowWriter(fps=10))

