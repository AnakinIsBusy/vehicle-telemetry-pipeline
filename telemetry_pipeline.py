import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt

# ==========================================
# 1. INGEST & TIME-SYNCHRONIZE SENSOR LOGS
# ==========================================
DATA_DIR = "data"
TARGET_FS = 50.0  # Resample all sensors to a unified 50 Hz grid

loc = pd.read_csv(f"{DATA_DIR}/Location.csv")
acc = pd.read_csv(f"{DATA_DIR}/Accelerometer.csv")
grav = pd.read_csv(f"{DATA_DIR}/Gravity.csv")
gyro = pd.read_csv(f"{DATA_DIR}/Gyroscope.csv")

# Determine overlapping time window across all sensors
t_start = max(loc["seconds_elapsed"].min(), acc["seconds_elapsed"].min(),
              grav["seconds_elapsed"].min(), gyro["seconds_elapsed"].min())
t_end = min(loc["seconds_elapsed"].max(), acc["seconds_elapsed"].max(),
            grav["seconds_elapsed"].max(), gyro["seconds_elapsed"].max())

t_grid = np.arange(t_start, t_end, 1.0 / TARGET_FS)

def resample_df(df, cols, time_grid):
    df = df.sort_values("seconds_elapsed").drop_duplicates("seconds_elapsed")
    out = {}
    for c in cols:
        out[c] = np.interp(time_grid, df["seconds_elapsed"].values, df[c].values)
    return pd.DataFrame(out)

df_loc = resample_df(loc, ["latitude", "longitude", "speed", "altitude"], t_grid)
df_acc = resample_df(acc, ["x", "y", "z"], t_grid)
df_grav = resample_df(grav, ["x", "y", "z"], t_grid)
df_gyro = resample_df(gyro, ["x", "y", "z"], t_grid)

df = pd.DataFrame({"time": t_grid})
df["lat"] = df_loc["latitude"]
df["lon"] = df_loc["longitude"]
df["speed_ms"] = np.clip(df_loc["speed"], 0, None)
df["speed_kmh"] = df["speed_ms"] * 3.6
df["altitude"] = df_loc["altitude"]

# Compute cumulative distance (F1 standard X-axis)
dt = 1.0 / TARGET_FS
df["distance_m"] = np.cumsum(df["speed_ms"] * dt)

# ==========================================
# 2. SIGNAL FILTERING (BUTTERWORTH LOW-PASS)
# ==========================================
def lowpass_filter(signal, cutoff=3.0, fs=TARGET_FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, signal)

for axis in ["x", "y", "z"]:
    df[f"acc_{axis}"] = lowpass_filter(df_acc[axis].values)
    df[f"grav_{axis}"] = lowpass_filter(df_grav[axis].values, cutoff=1.0)
    df[f"gyro_{axis}"] = lowpass_filter(df_gyro[axis].values)

# ==========================================
# 3. CHASSIS COORDINATE ALIGNMENT & MATH CHANNELS
# ==========================================
# Step A: Level the phone's frame using the Gravity vector (align Z with Earth vertical)
g_vec = df[["grav_x", "grav_y", "grav_z"]].values
g_norm = g_vec / np.linalg.norm(g_vec, axis=1, keepdims=True)

a_vec = df[["acc_x", "acc_y", "acc_z"]].values
# Vertical acceleration (heave) along true vertical
a_vert = np.sum(a_vec * g_norm, axis=1)
# Horizontal acceleration vector projected onto the road plane
a_horiz = a_vec - (a_vert[:, None] * g_norm)

# Step B: Separate Longitudinal (Throttle/Brake) vs Lateral (Cornering) G
# Using GPS-derived longitudinal acceleration to resolve vehicle heading on the horizontal plane
a_long_gps = lowpass_filter(np.gradient(df["speed_ms"].values, dt), cutoff=1.5)
a_horiz_mag = np.linalg.norm(a_horiz, axis=1)

# Lateral G magnitude derived from horizontal resultant minus longitudinal component
a_lat_mag = np.sqrt(np.clip(a_horiz_mag**2 - a_long_gps**2, 0, None))

# Assign sign to Lateral G using Yaw Rate (gyro projection onto gravity axis)
yaw_rate_rads = np.sum(df[["gyro_x", "gyro_y", "gyro_z"]].values * g_norm, axis=1)
df["yaw_rate_degs"] = np.degrees(yaw_rate_rads)

G = 9.80665
df["g_long"] = a_long_gps / G
df["g_lat"] = (a_lat_mag * np.sign(yaw_rate_rads)) / G
df["g_combined"] = np.sqrt(df["g_long"]**2 + df["g_lat"]**2)

# Convert Lat/Lon to local planar meters (X, Y) for accurate track map aspect ratio
R_EARTH = 6371000.0
lat0 = np.radians(df["lat"].iloc[0])
df["track_x"] = R_EARTH * np.radians(df["lon"] - df["lon"].iloc[0]) * np.cos(lat0)
df["track_y"] = R_EARTH * np.radians(df["lat"] - df["lat"].iloc[0])

# ==========================================
# 4. GENERATE F1 / MOTEC STYLE TELEMETRY SHEET
# ==========================================
plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 10), dpi=200)
gs = gridspec.GridSpec(4, 3, width_ratios=[1.1, 2.2, 1.0], hspace=0.25, wspace=0.25)

# --- Panel 1: GPS Circuit Map Colored by Speed ---
ax_map = fig.add_subplot(gs[:2, 0])
sc = ax_map.scatter(df["track_x"], df["track_y"], c=df["speed_kmh"], cmap="plasma", s=4)
ax_map.plot(df["track_x"].iloc[0], df["track_y"].iloc[0], "go", markersize=8, label="Start")
ax_map.plot(df["track_x"].iloc[-1], df["track_y"].iloc[-1], "rs", markersize=8, label="Finish")
ax_map.set_aspect("equal", "datalim")
ax_map.set_title("GPS Track Map (Speed Gradient)", fontsize=11, fontweight="bold")
ax_map.set_xlabel("East-West (m)")
ax_map.set_ylabel("North-South (m)")
ax_map.legend(loc="upper right", fontsize=8)
plt.colorbar(sc, ax=ax_map, label="Speed (km/h)", fraction=0.046, pad=0.04)

# --- Panel 2: G-G Friction Circle ---
ax_gg = fig.add_subplot(gs[2:, 0])
sc_gg = ax_gg.scatter(df["g_lat"], df["g_long"], c=df["speed_kmh"], cmap="viridis", alpha=0.4, s=5)
for radius in [0.25, 0.5, 0.75, 1.0]:
    circle = plt.Circle((0, 0), radius, color="gray", fill=False, linestyle="--", alpha=0.4)
    ax_gg.add_patch(circle)
ax_gg.axhline(0, color="white", lw=0.7, alpha=0.5)
ax_gg.axvline(0, color="white", lw=0.7, alpha=0.5)
max_g = max(0.6, df["g_combined"].max() * 1.15)
ax_gg.set_xlim(-max_g, max_g)
ax_gg.set_ylim(-max_g, max_g)
ax_gg.set_aspect("equal")
ax_gg.set_title("G-G Friction Circle (Traction Envelope)", fontsize=11, fontweight="bold")
ax_gg.set_xlabel("Lateral G (- Left / + Right)")
ax_gg.set_ylabel("Longitudinal G (- Brake / + Accel)")

# --- Panel 3: Distance-Synced Telemetry Traces ---
dist = df["distance_m"]

ax_spd = fig.add_subplot(gs[0, 1:])
ax_spd.plot(dist, df["speed_kmh"], color="#00E5FF", lw=1.5)
ax_spd.fill_between(dist, df["speed_kmh"], color="#00E5FF", alpha=0.1)
ax_spd.set_ylabel("Speed\n(km/h)", fontweight="bold")
ax_spd.set_title("F1-Style Telemetry Trace (Distance Domain)", fontsize=12, fontweight="bold")
ax_spd.grid(True, alpha=0.2)

ax_glong = fig.add_subplot(gs[1, 1:], sharex=ax_spd)
ax_glong.plot(dist, df["g_long"], color="#76FF03", lw=1.2)
ax_glong.fill_between(dist, 0, df["g_long"], where=(df["g_long"] < 0), color="#FF1744", alpha=0.4, label="Braking")
ax_glong.fill_between(dist, 0, df["g_long"], where=(df["g_long"] >= 0), color="#76FF03", alpha=0.3, label="Accel")
ax_glong.axhline(0, color="white", lw=0.6, alpha=0.5)
ax_glong.set_ylabel("Long. G\n(Brake/Accel)", fontweight="bold")
ax_glong.legend(loc="upper right", fontsize=8)
ax_glong.grid(True, alpha=0.2)

ax_glat = fig.add_subplot(gs[2, 1:], sharex=ax_spd)
ax_glat.plot(dist, df["g_lat"], color="#FFEA00", lw=1.2)
ax_glat.axhline(0, color="white", lw=0.6, alpha=0.5)
ax_glat.set_ylabel("Lateral G\n(Cornering)", fontweight="bold")
ax_glat.grid(True, alpha=0.2)

ax_yaw = fig.add_subplot(gs[3, 1:], sharex=ax_spd)
ax_yaw.plot(dist, df["yaw_rate_degs"], color="#E040FB", lw=1.2)
ax_yaw.axhline(0, color="white", lw=0.6, alpha=0.5)
ax_yaw.set_ylabel("Yaw Rate\n(deg/s)", fontweight="bold")
ax_yaw.set_xlabel("Distance (m)", fontweight="bold")
ax_yaw.grid(True, alpha=0.2)

plt.suptitle("VEHICLE TELEMETRY & DYNAMICS REPORT", fontsize=15, fontweight="bold", y=0.98)
plt.savefig("telemetry_report.png", bbox_inches="tight")
print(f"Done! Peak Speed: {df['speed_kmh'].max():.1f} km/h | Peak Braking: {df['g_long'].min():.2f} G | Peak Cornering: {df['g_lat'].abs().max():.2f} G")
