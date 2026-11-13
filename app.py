import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(layout="wide", page_title="Motorsport Telemetry Studio")

# ==========================================
# SIDEBAR CONTROLS (LIVE ENGINEERING TUNING)
# ==========================================
st.sidebar.header("Signal Processing & Sector")
cutoff_hz = st.sidebar.slider("Butterworth Low-Pass Cutoff (Hz)", 0.5, 10.0, 3.0, 0.5)
target_fs = 25.0

@st.cache_data
def load_raw_data(data_dir="data"):
    loc = pd.read_csv(f"{data_dir}/Location.csv")
    acc = pd.read_csv(f"{data_dir}/Accelerometer.csv")
    grav = pd.read_csv(f"{data_dir}/Gravity.csv")
    gyro = pd.read_csv(f"{data_dir}/Gyroscope.csv")

    t_start = max(loc["seconds_elapsed"].min(), acc["seconds_elapsed"].min(),
                  grav["seconds_elapsed"].min(), gyro["seconds_elapsed"].min())
    t_end = min(loc["seconds_elapsed"].max(), acc["seconds_elapsed"].max(),
                grav["seconds_elapsed"].max(), gyro["seconds_elapsed"].max())

    t_grid = np.arange(t_start, t_end, 1.0 / target_fs)

    def resample_df(df, cols):
        df = df.sort_values("seconds_elapsed").drop_duplicates("seconds_elapsed")
        return pd.DataFrame({c: np.interp(t_grid, df["seconds_elapsed"].values, df[c].values) for c in cols})

    df_loc = resample_df(loc, ["latitude", "longitude", "speed", "altitude"])
    df_acc = resample_df(acc, ["x", "y", "z"])
    df_grav = resample_df(grav, ["x", "y", "z"])
    df_gyro = resample_df(gyro, ["x", "y", "z"])

    base = pd.DataFrame({"time": np.round(t_grid - t_start, 2)})
    base["lat"] = df_loc["latitude"]
    base["lon"] = df_loc["longitude"]
    base["speed_ms"] = np.clip(df_loc["speed"], 0, None)
    base["speed_kmh"] = base["speed_ms"] * 3.6
    dt = 1.0 / target_fs
    base["distance_m"] = np.cumsum(base["speed_ms"] * dt)

    for prefix, src in [("acc", df_acc), ("grav", df_grav), ("gyro", df_gyro)]:
        for ax in ["x", "y", "z"]:
            base[f"{prefix}_{ax}"] = src[ax].values

    R_EARTH = 6371000.0
    lat0 = np.radians(base["lat"].iloc[0])
    base["track_x"] = R_EARTH * np.radians(base["lon"] - base["lon"].iloc[0]) * np.cos(lat0)
    base["track_y"] = R_EARTH * np.radians(base["lat"] - base["lat"].iloc[0])
    return base, dt

df, dt = load_raw_data()

def lowpass_filter(signal, cutoff, fs=target_fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, signal)

for ax in ["x", "y", "z"]:
    df[f"acc_{ax}_f"] = lowpass_filter(df[f"acc_{ax}"].values, cutoff_hz)
    df[f"grav_{ax}_f"] = lowpass_filter(df[f"grav_{ax}"].values, 1.0)
    df[f"gyro_{ax}_f"] = lowpass_filter(df[f"gyro_{ax}"].values, cutoff_hz)

# Gravity alignment & Math channels
g_vec = df[["grav_x_f", "grav_y_f", "grav_z_f"]].values
g_norm = g_vec / np.linalg.norm(g_vec, axis=1, keepdims=True)
a_vec = df[["acc_x_f", "acc_y_f", "acc_z_f"]].values
a_vert = np.sum(a_vec * g_norm, axis=1)
a_horiz = a_vec - (a_vert[:, None] * g_norm)

a_long_gps = lowpass_filter(np.gradient(df["speed_ms"].values, dt), min(cutoff_hz, 2.0))
a_horiz_mag = np.linalg.norm(a_horiz, axis=1)
a_lat_mag = np.sqrt(np.clip(a_horiz_mag**2 - a_long_gps**2, 0, None))

yaw_rate_rads = np.sum(df[["gyro_x_f", "gyro_y_f", "gyro_z_f"]].values * g_norm, axis=1)
df["yaw_rate_degs"] = np.degrees(yaw_rate_rads)

G = 9.80665
df["g_long"] = a_long_gps / G
df["g_lat"] = (a_lat_mag * np.sign(yaw_rate_rads)) / G

# Distance Sector Filter
max_dist = float(df["distance_m"].max())
dist_range = st.sidebar.slider("Inspect Distance Window (m)", 0.0, max_dist, (0.0, max_dist), 10.0)
view = df[(df["distance_m"] >= dist_range[0]) & (df["distance_m"] <= dist_range[1])]

# Top KPI Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Peak Speed", f"{view['speed_kmh'].max():.1f} km/h")
c2.metric("Peak Braking", f"{view['g_long'].min():.2f} G")
c3.metric("Peak Lateral G", f"{view['g_lat'].abs().max():.2f} G")
c4.metric("Sector Distance", f"{dist_range[1] - dist_range[0]:.0f} m")

# Build Synchronized Plotly Workspace
fig = make_subplots(
    rows=4, cols=2,
    column_widths=[0.33, 0.67],
    specs=[
        [{"type": "xy", "rowspan": 2}, {"type": "xy"}],
        [None,                         {"type": "xy"}],
        [{"type": "xy", "rowspan": 2}, {"type": "xy"}],
        [None,                         {"type": "xy"}]
    ],
    subplot_titles=(
        "GPS Circuit Map (Highlighted Sector)", "Speed Trace (km/h)",
        "Longitudinal G (Braking / Accel)",
        "G-G Friction Circle", "Lateral G (Cornering)",
        "Yaw Rate (deg/s)"
    ),
    vertical_spacing=0.07,
    horizontal_spacing=0.06
)

# Full track faint outline + active sector highlight
fig.add_trace(go.Scatter(x=df["track_x"], y=df["track_y"], mode="lines", line=dict(color="rgba(255,255,255,0.15)", width=2), hoverinfo="skip"), row=1, col=1)
fig.add_trace(go.Scatter(
    x=view["track_x"], y=view["track_y"], mode="markers",
    marker=dict(size=5, color=view["speed_kmh"], colorscale="Plasma", colorbar=dict(title="km/h", x=0.28, y=0.78, len=0.4, thickness=10)),
    customdata=np.stack((view["distance_m"], view["speed_kmh"]), axis=-1),
    hovertemplate="Dist: %{customdata[0]:.0f} m<br>Speed: %{customdata[1]:.1f} km/h<extra></extra>"
), row=1, col=1)

# G-G Diagram
fig.add_trace(go.Scatter(
    x=view["g_lat"], y=view["g_long"], mode="markers",
    marker=dict(size=4, color=view["speed_kmh"], colorscale="Viridis", opacity=0.6),
    hovertemplate="Lat G: %{x:.2f}G<br>Long G: %{y:.2f}G<extra></extra>"
), row=3, col=1)

theta = np.linspace(0, 2*np.pi, 100)
for r in [0.25, 0.5, 0.75, 1.0]:
    fig.add_trace(go.Scatter(x=r*np.cos(theta), y=r*np.sin(theta), mode="lines", line=dict(color="gray", width=1, dash="dot"), hoverinfo="skip"), row=3, col=1)

# Synchronized Traces
fig.add_trace(go.Scatter(x=view["distance_m"], y=view["speed_kmh"], mode="lines", line=dict(color="#00E5FF", width=2), fill="tozeroy", fillcolor="rgba(0, 229, 255, 0.1)", name="Speed"), row=1, col=2)
fig.add_trace(go.Scatter(x=view["distance_m"], y=view["g_long"], mode="lines", line=dict(color="#76FF03", width=1.5), fill="tozeroy", fillcolor="rgba(118, 255, 3, 0.12)", name="Long G"), row=2, col=2)
fig.add_trace(go.Scatter(x=view["distance_m"], y=view["g_lat"], mode="lines", line=dict(color="#FFEA00", width=1.5), name="Lat G"), row=3, col=2)
fig.add_trace(go.Scatter(x=view["distance_m"], y=view["yaw_rate_degs"], mode="lines", line=dict(color="#E040FB", width=1.5), name="Yaw Rate"), row=4, col=2)

fig.update_xaxes(matches="x2", row=2, col=2)
fig.update_xaxes(matches="x2", row=3, col=2)
fig.update_xaxes(matches="x2", row=4, col=2, title_text="Cumulative Distance (m)")
fig.update_yaxes(scaleanchor="x1", scaleratio=1, row=1, col=1)
fig.update_yaxes(scaleanchor="x3", scaleratio=1, row=3, col=1)
fig.update_xaxes(title_text="Lateral G", row=3, col=1)
fig.update_yaxes(title_text="Longitudinal G", row=3, col=1)

fig.update_layout(template="plotly_dark", height=820, showlegend=False, hovermode="x unified", margin=dict(t=40, b=20))
st.plotly_chart(fig, width="stretch")
