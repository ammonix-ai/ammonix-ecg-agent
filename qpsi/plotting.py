"""
Visualization -- all matplotlib / plotly plotting for the QPSI pipeline.

Source: Cell 2 (R-peak and baseline plots), Cell 8 (angle color mapping, wave overlay)
Depends on: constants (LOG_EPS, etc.); matplotlib, plotly are OPTIONAL
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("qpsi")

# ---------------------------------------------------------------------------
# Optional plotting library imports
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import FuncFormatter

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants imported from qpsi.constants
# ---------------------------------------------------------------------------
from qpsi.constants import (
    LOG_EPS,
    QRS_FRAC_MS,
    ST_FRAC,
    EARLY_P_MS,
    NORMAL_P_MS,
)

# ---------------------------------------------------------------------------
# Type alias (from Cell 8)
# ---------------------------------------------------------------------------
Angle = Union[float, np.ndarray]


# =========================================================================
# 1.  Angle -> CMYK -> RGB mapping  (Cell 8)
# =========================================================================

def angle_to_CMYK(angle_xy: Angle) -> Tuple[float, float, float, float]:
    """Return CMYK for **one** angle.  Reject vectors.

    Parameters
    ----------
    angle_xy : float or scalar np.ndarray
        A single angle in radians.

    Returns
    -------
    (c, m, y, k) : tuple of four floats
        CMYK colour values in [0, 1].

    Raises
    ------
    ValueError
        If *angle_xy* is an array with more than one element.
    """
    a = np.asarray(angle_xy)
    if a.size != 1:
        raise ValueError(
            "angle_to_CMYK expects a scalar angle; got shape "
            f"{a.shape}. Use a loop or vectorised colour map."
        )
    x = float(a)                     # scalar
    c = 0.5                          # sin(0)=0 -> constant
    m = 0.5 * (1.0 + math.cos(x))
    y = 0.5 * (1.0 + math.sin(x))
    k = 0.0                          # cos(0)=1
    return c, m, y, k


def cmyk_to_rgb(
    c: float, m: float, y: float, k: float,
) -> Tuple[float, float, float]:
    """Convert CMYK to RGB.

    Parameters
    ----------
    c, m, y, k : float
        CMYK components in [0, 1].

    Returns
    -------
    (r, g, b) : tuple of three floats
        RGB components in [0, 1].
    """
    return (1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)


# =========================================================================
# 2.  Magnitude and angle computation  (Cell 8)
# =========================================================================

def compute_magnitude_and_angles(
    v_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute magnitude and angles from 3-D vector traces.

    Parameters
    ----------
    v_xyz : np.ndarray, shape (3, N)
        Three-component vector signal (x, y, z) over *N* samples.

    Returns
    -------
    magnitude : np.ndarray, shape (N,)
        Euclidean norm at each sample.
    angles_xy : np.ndarray, shape (N,)
        Angle in the XY (frontal) plane, range [-pi, pi].
        0 rad => patient's left (Lead I positive),
        +pi/2 => downward (aVF positive).
    angles_xz : np.ndarray, shape (N,)
        Angle in the XZ (transverse) plane, range [-pi, pi].
    """
    vx = v_xyz[0, :]   # +vx = to patient's left
    vy = v_xyz[1, :]   # +vy = downward
    vz = v_xyz[2, :]

    magnitude = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    # 1) Get standard arctan2 in radians:
    #    By default: 0 rad => x>0 => "right", +pi/2 => "up"
    angles_xy = np.arctan2(vy, vx)
    angles_xz = np.arctan2(vz, vx)

    return magnitude, angles_xy, angles_xz


# =========================================================================
# 3.  Plotly: R-peak detection debug plot  (Cell 2)
# =========================================================================

def plot_r_peak_detection_plotly(
    env: np.ndarray,
    dv: np.ndarray,
    fs: int,
    pk: np.ndarray,
    fin_pk: np.ndarray,
    top_percent: float = 40.0,
    height: int = 350,
    width: int = 900,
    *,
    plot_enabled: bool = False,
) -> None:
    """Interactive Plotly version of the R-peak detector plot.

    Traces shown:

    * 'energy'            = *env*
    * 'dv (norm)'         = derivative, scaled to +/-1 and shifted down
    * orange open circles = all amplitude candidates (*pk*)
    * blue diamonds       = peaks that survived the steepness gate (*fin_pk*)

    Parameters
    ----------
    env : np.ndarray
        1-D energy / envelope signal.
    dv : np.ndarray
        First derivative of *env* (same length).
    fs : int
        Sampling frequency in Hz.
    pk : np.ndarray
        Indices of all amplitude candidates.
    fin_pk : np.ndarray
        Indices of the peaks kept after steepness gating.
    top_percent : float
        Steepness gate percentage (for annotation only).
    height, width : int
        Figure dimensions in pixels.
    plot_enabled : bool
        When *False* (default) the function returns immediately without
        creating a figure.

    Returns
    -------
    None
    """
    if not plot_enabled:
        return None
    if not _PLOTLY_AVAILABLE:
        logger.warning(
            "plot_r_peak_detection_plotly: plotly not installed -- skipping."
        )
        return None

    t = np.arange(env.size) / fs

    # normalise dv to +/-1 and offset it downward so it doesn't overlap env
    dv_norm = dv / (np.max(np.abs(dv)) + 1e-9)
    y_off = -0.15 * env.max()        # shift derivative trace downward
    dv_disp = dv_norm * 0.1 * env.max() + y_off

    fig = go.Figure()

    # -- envelope trace -------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=t, y=env,
        mode="lines",
        line=dict(width=1, color="royalblue"),
        name="energy",
    ))

    # -- dv trace (toggleable) ------------------------------------------------
    fig.add_trace(go.Scatter(
        x=t, y=dv_disp,
        mode="lines",
        line=dict(width=1, color="green"),
        name="dv (norm, offset)",
    ))

    # -- candidate peaks ------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=t[pk], y=env[pk],
        mode="markers",
        marker=dict(
            symbol="circle-open", color="orange", size=8,
            line=dict(width=1),
        ),
        name="candidates",
    ))

    # -- accepted peaks -------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=t[fin_pk], y=env[fin_pk],
        mode="markers",
        marker=dict(
            symbol="diamond", color="blue", size=8,
            line=dict(color="black", width=1),
        ),
        name="selected",
    ))

    fig.update_xaxes(title_text="Time [s]")
    fig.update_yaxes(title_text="Amplitude (a.u.)")

    fig.show()


# =========================================================================
# 4.  Plotly: Two-anchor baseline correction visualisation  (Cell 2)
# =========================================================================

def plot_baseline_twoanchor_plotly(
    ecg_filt: np.ndarray,
    ecg_bs: np.ndarray,
    bs_curve: np.ndarray,
    anchors_x: list[list[int]],
    anchors_y: list[list[float]],
    r_peaks: np.ndarray,
    fs: int = 500,
    zoom_rr: tuple[int, int] = (1, 2),
    *,
    plot_enabled: bool = False,
) -> None:
    """Interactive Plotly version of the two-anchor baseline plot.

    Shows filtered trace (black), baseline-subtracted trace (green),
    estimated baseline curve (red), and anchor points (crimson crosses)
    for each lead.  A secondary time axis (seconds) is overlaid on top.

    Parameters
    ----------
    ecg_filt : np.ndarray, shape (n_leads, N)
        Band-pass filtered ECG.
    ecg_bs : np.ndarray, shape (n_leads, N)
        Baseline-subtracted ECG.
    bs_curve : np.ndarray, shape (n_leads, N)
        Estimated baseline curve.
    anchors_x : list[list[int]]
        Per-lead sample indices of the baseline anchors.
    anchors_y : list[list[float]]
        Per-lead amplitudes at the baseline anchors.
    r_peaks : np.ndarray
        R-peak sample indices.
    fs : int
        Sampling frequency in Hz.
    zoom_rr : (int, int)
        Tuple *(i_start, i_end)* specifying which RR-intervals (0-based)
        to display initially.  Default ``(1, 2)`` shows the 2nd and 3rd RR.
    plot_enabled : bool
        When *False* (default) the function returns immediately.

    Returns
    -------
    None
    """
    if not plot_enabled:
        return None
    if not _PLOTLY_AVAILABLE:
        logger.warning(
            "plot_baseline_twoanchor_plotly: plotly not installed -- skipping."
        )
        return None

    n_leads, N = ecg_filt.shape
    lead_names = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ][:n_leads]

    # ---------------------------------------------------------------- default zoom window
    rr_left = r_peaks[zoom_rr[0]]
    rr_right = (
        r_peaks[zoom_rr[1] + 1]
        if (zoom_rr[1] + 1) < len(r_peaks)
        else N - 1
    )
    x0, x1 = rr_left, rr_right

    # ---------------------------------------------------------------- figure grid
    fig = make_subplots(
        rows=n_leads, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, subplot_titles=lead_names,
    )

    # ---------------------------------------------------------------- traces per lead
    for ld in range(n_leads):
        x_vals = np.arange(N)                     # sample index

        # filtered trace (black)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=ecg_filt[ld],
                line=dict(width=0.7, color="black"),
                name="filtered" if ld == 0 else None,
                showlegend=(ld == 0),
                hovertemplate=(
                    "sample=%{x}<br>time=%{customdata:.3f}s<br>amp=%{y:.3f}"
                ),
                customdata=x_vals / fs,            # seconds for hover
            ),
            row=ld + 1, col=1,
        )

        # baseline-subtracted (green)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=ecg_bs[ld],
                line=dict(width=0.5, color="green"),
                name="bs-sub" if ld == 0 else None,
                showlegend=(ld == 0),
                hovertemplate=(
                    "sample=%{x}<br>time=%{customdata:.3f}s<br>amp=%{y:.3f}"
                ),
                customdata=x_vals / fs,
            ),
            row=ld + 1, col=1,
        )

        # baseline curve (red)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=bs_curve[ld],
                line=dict(width=1, color="red"),
                name="baseline" if ld == 0 else None,
                showlegend=(ld == 0),
                hovertemplate=(
                    "sample=%{x}<br>time=%{customdata:.3f}s<br>amp=%{y:.3f}"
                ),
                customdata=x_vals / fs,
            ),
            row=ld + 1, col=1,
        )

        # anchor points (crosses)
        fig.add_trace(
            go.Scatter(
                x=anchors_x[ld],
                y=anchors_y[ld],
                mode="markers",
                marker=dict(symbol="x", color="crimson", size=6),
                name="anchors" if ld == 0 else None,
                showlegend=(ld == 0),
                hovertemplate=(
                    "sample=%{x}<br>time=%{customdata:.3f}s<br>amp=%{y:.3f}"
                ),
                customdata=np.array(anchors_x[ld]) / fs,
            ),
            row=ld + 1, col=1,
        )

    # ---------------------------------------------------------------- two overlaid X-axes
    # 1) primary bottom axis -> sample index
    fig.update_xaxes(
        title_text="sample index",
        range=[x0, x1],
        row=n_leads, col=1,
    )

    # 2) secondary top axis -> time (s)
    tickvals = np.linspace(x0, x1, 8)
    ticktext = [f"{v / fs:.2f}" for v in tickvals]

    fig.update_layout(
        xaxis2=dict(                       # overlay on xaxis (Plotly default id)
            overlaying="x",
            side="top",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            title="time (s)",
        ),
    )

    # ---------------------------------------------------------------- layout tweaks
    fig.update_layout(
        height=int(220 * n_leads),
        width=2000,
        title_text="Two-anchor baseline (interactive)",
        hovermode="x unified",
        margin=dict(t=80, b=40),
    )
    fig.update_yaxes(title_text="mV")

    fig.show()


# =========================================================================
# 5.  Matplotlib: Main dot-plot with Sigma-fit overlay  (Cell 8)
# =========================================================================

def plot_rr_with_waves_main(
    t_array: np.ndarray,
    mag_data: np.ndarray,
    angle: np.ndarray,
    wave_list: List[dict],
    *,
    rr_ms: float,
    amplitude_threshold: float = 0.04,
    offset_ms: float = 50.0,
    title: str = "RR-interval => waves",
    recording_name: str = "(unnamed)",
    hide_xaxis: bool = False,
    plot_enabled: bool = False,
) -> Any:
    """Main dot-plot with Sigma-fit overlay.

    Each sample is drawn as a coloured dot (CMYK-mapped from its frontal
    angle); Gaussian-fit wave envelopes and a total Sigma-fit line are
    overlaid.  Background bands shade QRS / ST / T / P regions.

    Parameters
    ----------
    t_array : np.ndarray
        1-D time axis in milliseconds (0 = R-peak).
    mag_data : np.ndarray
        1-D magnitude |v| at each sample.
    angle : np.ndarray
        1-D frontal-plane angle (radians) at each sample.
    wave_list : list of dict
        Each dict describes one wave component and must contain at least
        ``'t'``, ``'fit_amp'``, ``'peak_angle'``, and optionally ``'label'``.
    rr_ms : float
        RR-interval length in milliseconds.
    amplitude_threshold : float
        Minimum amplitude for log-scale display.
    offset_ms : float
        Guard band in ms between QRS and P-side.
    title : str
        Figure title.
    recording_name : str
        Recording identifier appended to the title.
    hide_xaxis : bool
        If *True*, suppress the X axis label and ticks.
    plot_enabled : bool
        When *False* (default) the function returns ``None`` immediately.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object, or ``None`` if plotting was skipped.
    """
    if not plot_enabled:
        return None
    if not _MPL_AVAILABLE:
        logger.warning(
            "plot_rr_with_waves_main: matplotlib not installed -- skipping."
        )
        return None

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(figsize=(12, 4))
    if recording_name.lower() in title.lower():
        full_title = title
    else:
        full_title = f"{title}   [{recording_name}]"

    ax.set_title(full_title, pad=14, fontsize=13)
    # Make main axis transparent so background shading is visible
    ax.patch.set_alpha(0.0)
    ax.set_zorder(1)

    # ---------------------------------------------------------------- shading (background bands)
    # draw shading on main ax before data so it's behind points/lines
    left, right = -rr_ms / 2 + offset_ms, rr_ms / 2 + offset_ms
    # QRS / ST / T region
    ax.axvspan(
        -offset_ms, QRS_FRAC_MS,
        facecolor="salmon", alpha=0.20, zorder=0,
    )
    ax.axvspan(
        QRS_FRAC_MS, rr_ms * ST_FRAC,
        facecolor="skyblue", alpha=0.20, zorder=0,
    )
    ax.axvspan(
        rr_ms * ST_FRAC, right,
        facecolor="lightgray", alpha=0.20, zorder=0,
    )
    # P-wave region
    ax.axvspan(
        left, EARLY_P_MS,
        facecolor="orange", alpha=0.20, zorder=0,
    )
    ax.axvspan(
        EARLY_P_MS, NORMAL_P_MS,
        facecolor="palegreen", alpha=0.20, zorder=0,
    )
    ax.axvspan(
        NORMAL_P_MS, -offset_ms,
        facecolor="khaki", alpha=0.20, zorder=0,
    )

    # ---------------------------------------------------------------- points
    ln_mag = np.log(mag_data + amplitude_threshold + LOG_EPS)
    for i, t_val in enumerate(t_array):
        # pick the *scalar* angle that belongs to this time-sample
        c, m, y, k = angle_to_CMYK(angle[i])        # scalar, not vector
        ax.plot(
            t_val, ln_mag[i],
            marker="o", ms=4,
            mec="none",
            mfc=cmyk_to_rgb(c, m, y, k),
            alpha=0.9,
            zorder=2,
        )

    # ---------------------------------------------------------------- Sigma-fit overlay
    fit_total: Optional[np.ndarray] = None
    if wave_list:
        fit_total = np.zeros_like(t_array, dtype=float)
        for w in wave_list:
            fit_total += w["fit_amp"]
        ax.plot(
            t_array,
            np.log(fit_total + amplitude_threshold + LOG_EPS),
            lw=0.5, color="k", label="Sigma fit (model)", zorder=3,
        )

    # -------------------------------------------------------- per-wave envelopes
    used_labels: set = set()
    for w in wave_list:
        t_w = w["t"]
        amp = w["fit_amp"]
        ln_amp = np.log(amp + amplitude_threshold + LOG_EPS)

        label = w.get("label")
        if label in used_labels:
            label = None
        else:
            used_labels.add(label)

        rgb = cmyk_to_rgb(*angle_to_CMYK(w["peak_angle"]))
        ax.plot(t_w, ln_amp, lw=1.2, color=rgb, label=label, zorder=3)
        ax.fill_between(
            t_w, np.log(amplitude_threshold), ln_amp,
            where=amp > amplitude_threshold,
            color=rgb, alpha=0.25, zorder=2,
        )

    # ---------------------------------------------------------------- cosmetics
    ax.set_xlim(left, right)
    ax.set_ylabel("log |v|  (a.u.)")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6, prune="both"))
    ax.grid(ls=":", zorder=1)

    if hide_xaxis:
        ax.xaxis.set_visible(False)
    else:
        ax.set_xlabel("time (ms)   (0 ms = R-peak)")

    # y-range
    y_min = np.log(amplitude_threshold) - 0.5
    y_max = max(
        ln_mag.max(),
        (
            np.log(fit_total + amplitude_threshold + LOG_EPS).max()
            if fit_total is not None
            else ln_mag.max()
        ),
    ) + 0.5

    # ===== GUARD =========================================================
    # 1) replace non-finite values by a small default window
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = -1.0, 1.0           # fallback range (log-scale)

    # 2) make sure the window is >0 even if y_min ~ y_max
    if y_max - y_min < 1e-3:
        centre = 0.5 * (y_max + y_min)
        y_min, y_max = centre - 1.0, centre + 1.0
    # =====================================================================

    ax.set_ylim(y_min, y_max)

    # ---------------------------------------------------------------- scale bar (vertical at right)
    # colours for angles -180 ... +180 deg  ->  361 samples, 1 deg each
    grad_deg = np.linspace(-180.0, 180.0, 361)
    grad_rgb = np.array([
        cmyk_to_rgb(*angle_to_CMYK(np.radians(a)))
        for a in grad_deg
    ])                                                # shape 361x3
    grad_rgb = grad_rgb.reshape(361, 1, 3)            # make it vertical

    # [left, bottom, width, height] in *figure fraction* units
    bar_ax = fig.add_axes([0.88, 0.18, 0.02, 0.64])  # narrow bar at right
    bar_ax.imshow(
        grad_rgb, aspect="auto",
        origin="lower", extent=[0, 1, -180, 180],
    )
    bar_ax.set_xticks([])                             # no x-ticks

    def _mv_formatter(y_val: float, _pos: Any) -> str:
        mv = np.exp(y_val) - amplitude_threshold - LOG_EPS   # inverse of log()
        return f"{mv:.2f}"

    ax.yaxis.set_major_formatter(FuncFormatter(_mv_formatter))
    ax.set_ylabel("|v|  (mV)")
    bar_ax.yaxis.set_ticks([-180, -90, 0, 90, 180])
    bar_ax.yaxis.set_tick_params(labelsize=7)

    plt.show()
    return fig
