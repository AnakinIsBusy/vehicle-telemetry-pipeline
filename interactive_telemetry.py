import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. INGEST, SYNC & FILTER SENSOR LOGS
# ==========================================
DATA_DIR = "data"
TARGET_FS = 25.0  # 25 Hz keeps interactive HTML fast and responsive

loc = pd.read_csv(f"{DATA_DIR}/Location.csv")
acc = pd.read_csv(f"{DATA_DIR}/Accelerometer.csv")
grav = pd.read_csv(f"{DATA_DIR}/Gravity.csv")
gyro = pd.read_csv(f"{DATA_DIR}/Gyroscope.csv")

t_start = max(loc["seconds_elapsed"].min(), acc["seconds_elapsed"].min(),
              grav["seconds_elapsed"].min(), gyro["seconds_elapsed"].min())
t_end = min(loc["seconds_elapsed"].max(), acc["seconds_elapsed"].max(),
            grav["seconds_elapsed"].max(), gyro["seconds_elapsed"].max())

t_grid = np.arange(t_start, t_end, 1.0 / TARGET_FS)

def resample_df(df, cols, time_grid):
    df = df.sort_values("seconds_elapsed").drop_duplicates("seconds_elapsed")
    return pd.DataFrame({c: np.interp(time_grid, df["seconds_elapsed"].values, df[c].values) for c in cols})

df_loc = resample_df(loc, ["latitude", "longitude", "speed", "altitude"], t_grid)
df_acc = resample_df(acc, ["x", "y", "z"], t_grid)
df_grav = resample_df(grav, ["x", "y", "z"], t_grid)
df_gyro = resample_df(gyro, ["x", "y", "z"], t_grid)

df = pd.DataFrame({"time": np.round(t_grid - t_start, 2)})
df["lat"] = df_loc["latitude"]
df["lon"] = df_loc["longitude"]
df["speed_ms"] = np.clip(df_loc["speed"], 0, None)
df["speed_kmh"] = df["speed_ms"] * 3.6
dt = 1.0 / TARGET_FS
df["distance_m"] = np.cumsum(df["speed_ms"] * dt)

def lowpass_filter(signal, cutoff=3.0, fs=TARGET_FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, signal)

for axis in ["x", "y", "z"]:
    df[f"acc_{axis}"] = lowpass_filter(df_acc[axis].values)
    df[f"grav_{axis}"] = lowpass_filter(df_grav[axis].values, cutoff=1.0)
    df[f"gyro_{axis}"] = lowpass_filter(df_gyro[axis].values)

# Gravity alignment & Math channels
g_vec = df[["grav_x", "grav_y", "grav_z"]].values
g_norm = g_vec / np.linalg.norm(g_vec, axis=1, keepdims=True)
a_vec = df[["acc_x", "acc_y", "acc_z"]].values
a_vert = np.sum(a_vec * g_norm, axis=1)
a_horiz = a_vec - (a_vert[:, None] * g_norm)

a_long_gps = lowpass_filter(np.gradient(df["speed_ms"].values, dt), cutoff=1.5)
a_horiz_mag = np.linalg.norm(a_horiz, axis=1)
a_lat_mag = np.sqrt(np.clip(a_horiz_mag**2 - a_long_gps**2, 0, None))

yaw_rate_rads = np.sum(df[["gyro_x", "gyro_y", "gyro_z"]].values * g_norm, axis=1)
df["yaw_rate_degs"] = np.degrees(yaw_rate_rads)

G = 9.80665
df["g_long"] = a_long_gps / G
df["g_lat"] = (a_lat_mag * np.sign(yaw_rate_rads)) / G
df["g_combined"] = np.sqrt(df["g_long"]**2 + df["g_lat"]**2)

R_EARTH = 6371000.0
lat0 = np.radians(df["lat"].iloc[0])
df["track_x"] = R_EARTH * np.radians(df["lon"] - df["lon"].iloc[0]) * np.cos(lat0)
df["track_y"] = R_EARTH * np.radians(df["lat"] - df["lat"].iloc[0])

# ==========================================
# 2. BUILD INTERACTIVE MOTEC/F1 WORKSPACE
# ==========================================
fig = make_subplots(
    rows=4, cols=2,
    column_widths=[0.33, 0.67],
    row_heights=[0.25, 0.25, 0.25, 0.25],
    specs=[
        [{"type": "xy", "rowspan": 2}, {"type": "xy"}],
        [None,                         {"type": "xy"}],
        [{"type": "xy", "rowspan": 2}, {"type": "xy"}],
        [None,                         {"type": "xy"}]
    ],
    subplot_titles=(
        "GPS Circuit Map (Hover for Distance & Speed)", "Speed Trace (km/h)",
        "Longitudinal G (Braking / Accel)",
        "G-G Friction Circle (Traction Envelope)", "Lateral G (Cornering)",
        "Yaw Rate (deg/s)"
    ),
    vertical_spacing=0.07,
    horizontal_spacing=0.06
)

# 1. GPS Track Map (Top Left)
fig.add_trace(
    go.Scatter(
        x=df["track_x"], y=df["track_y"],
        mode="markers+lines",
        line=dict(color="rgba(255,255,255,0.2)", width=1),
        marker=dict(
            size=5, color=df["speed_kmh"], colorscale="Plasma",
            colorbar=dict(title="km/h", x=0.28, y=0.78, len=0.4, thickness=10)
        ),
        customdata=np.stack((df["distance_m"], df["speed_kmh"], df["g_long"], df["g_lat"]), axis=-1),
        hovertemplate="<b>Dist: %{customdata[0]:.0f} m</b><br>Speed: %{customdata[1]:.1f} km/h<br>Long G: %{customdata[2]:.2f}G<br>Lat G: %{customdata[3]:.2f}G<extra></extra>",
        name="GPS Track"
    ),
    row=1, col=1
)

# 2. G-G Friction Circle (Bottom Left)
fig.add_trace(
    go.Scatter(
        x=df["g_lat"], y=df["g_long"],
        mode="markers",
        marker=dict(size=4, color=df["speed_kmh"], colorscale="Viridis", opacity=0.6),
        customdata=np.stack((df["distance_m"], df["speed_kmh"]), axis=-1),
        hovertemplate="<b>Dist: %{customdata[0]:.0f} m</b><br>Lat G: %{x:.2f} G<br>Long G: %{y:.2f} G<br>Speed: %{customdata[1]:.1f} km/h<extra></extra>",
        name="G-G Envelope"
    ),
    row=3, col=1
)

# Add reference G-rings to G-G diagram
theta = np.linspace(0, 2*np.pi, 100)
for r in [0.25, 0.5, 0.75, 1.0]:
    fig.add_trace(
        go.Scatter(
            x=r*np.cos(theta), y=r*np.sin(theta),
            mode="lines", line=dict(color="gray", width=1, dash="dot"),
            hoverinfo="skip", showlegend=False
        ),
        row=3, col=1
    )

# 3. Synchronized Distance Traces (Right Column)
fig.add_trace(
    go.Scatter(
        x=df["distance_m"], y=df["speed_kmh"],
        mode="lines", line=dict(color="#00E5FF", width=2),
        fill="tozeroy", fillcolor="rgba(0, 229, 255, 0.1)",
        name="Speed", hovertemplate="%{y:.1f} km/h"
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=df["distance_m"], y=df["g_long"],
        mode="lines", line=dict(color="#76FF03", width=1.5),
        fill="tozeroy", fillcolor="rgba(118, 255, 3, 0.12)",
        name="Long G", hovertemplate="%{y:.2f} G"
    ),
    row=2, col=2
)

fig.add_trace(
    go.Scatter(
        x=df["distance_m"], y=df["g_lat"],
        mode="lines", line=dict(color="#FFEA00", width=1.5),
        name="Lat G", hovertemplate="%{y:.2f} G"
    ),
    row=3, col=2
)

fig.add_trace(
    go.Scatter(
        x=df["distance_m"], y=df["yaw_rate_degs"],
        mode="lines", line=dict(color="#E040FB", width=1.5),
        name="Yaw Rate", hovertemplate="%{y:.1f} deg/s"
    ),
    row=4, col=2
)

# Link all 4 right-side telemetry X-axes together for synchronized zoom & pan
fig.update_xaxes(matches="x2", row=2, col=2)
fig.update_xaxes(matches="x2", row=3, col=2)
fig.update_xaxes(matches="x2", row=4, col=2, title_text="Cumulative Distance (m)")

# Keep Track Map & G-G Circle aspect ratios 1:1
fig.update_yaxes(scaleanchor="x1", scaleratio=1, row=1, col=1)
fig.update_yaxes(scaleanchor="x3", scaleratio=1, row=3, col=1)
fig.update_xaxes(title_text="Lateral G (- Left / + Right)", row=3, col=1)
fig.update_yaxes(title_text="Longitudinal G (- Brake / + Accel)", row=3, col=1)

# Dark Motorsport Theme & Unified Crosshair
fig.update_layout(
    template="plotly_dark",
    title=dict(
        text=f"<b>INTERACTIVE VEHICLE TELEMETRY WORKSPACE</b>  |  Peak Speed: {df['speed_kmh'].max():.1f} km/h  |  Max Braking: {df['g_long'].min():.2f} G  |  Max Lat: {df['g_lat'].abs().max():.2f} G",
        font=dict(size=16)
    ),
    height=900,
    showlegend=False,
    hovermode="x unified"
)

# Add vertical spike line across all telemetry channels on hover
for r in [1, 2, 3, 4]:
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="white", spikethickness=1, row=r, col=2)

fig.write_html("telemetry_interactive.html", include_plotlyjs="cdn")
print("Saved interactive dashboard to telemetry_interactive.html!")
fig.show()


import streamlit as st
st.set_page_config(layout="wide", page_title="Apex Telemetry Studio")
st.plotly_chart(fig, use_container_width=True)
