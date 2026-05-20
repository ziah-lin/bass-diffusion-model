"""
Bass Diffusion Model — Implementation, Fitting, and Forecasting
================================================================

A clean, generic implementation of the Bass Diffusion Model for forecasting
adoption of new products and technologies.

The Bass Diffusion Model (Bass, 1969) decomposes adoption into two forces:
  - Innovation (p): people who adopt independently of social pressure
  - Imitation (q): people who adopt because others already have
  - Market potential (m): the total addressable population

This script:
  1. Implements the Bass equations from scratch (numpy)
  2. Fits the model to analogous historical cases (scipy.optimize)
  3. Forecasts adoption for a new product anchored on analog parameters
  4. Visualizes adoption curves, cumulative adoption, and parameter uncertainty

Note: This is a sanitized, generic reproduction of a methodology I used in a
client engagement. No client data, product specifics, or proprietary parameters
are included.

Author: Ziah Lin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 1. THE BASS MODEL — CORE EQUATIONS
# ---------------------------------------------------------------------------

def bass_adopters(t, p, q, m):
    """
    Number of new adopters at time t under the Bass Diffusion Model.

    Closed-form solution:
        n(t) = m * [(p + q)^2 / p] * exp(-(p+q)t) / (1 + (q/p) * exp(-(p+q)t))^2

    Parameters
    ----------
    t : array-like
        Time periods (e.g., years since launch)
    p : float
        Coefficient of innovation (typical range: 0.0001 to 0.03)
    q : float
        Coefficient of imitation (typical range: 0.1 to 0.6)
    m : float
        Market potential (total eventual adopters)

    Returns
    -------
    np.ndarray
        New adopters per period
    """
    t = np.asarray(t, dtype=float)
    numerator = m * ((p + q) ** 2 / p) * np.exp(-(p + q) * t)
    denominator = (1 + (q / p) * np.exp(-(p + q) * t)) ** 2
    return numerator / denominator


def bass_cumulative(t, p, q, m):
    """Cumulative adopters at time t."""
    t = np.asarray(t, dtype=float)
    numerator = 1 - np.exp(-(p + q) * t)
    denominator = 1 + (q / p) * np.exp(-(p + q) * t)
    return m * numerator / denominator


# ---------------------------------------------------------------------------
# 2. ANALOGOUS HISTORICAL CASES
# ---------------------------------------------------------------------------
# In a real engagement, these would be carefully selected products that
# share key attributes with the target product: regulatory pathway, buyer
# persona, category maturity, network effects, etc.
#
# Here we use three synthetic but realistic analog cases that illustrate
# the range of parameter values seen across innovation diffusion studies.
# ---------------------------------------------------------------------------

@dataclass
class AnalogCase:
    name: str
    p: float            # innovation coefficient
    q: float            # imitation coefficient
    description: str    # category/notes (one line)


ANALOG_CASES = [
    AnalogCase(
        name="Analog A — Slow, deliberate adoption",
        p=0.003, q=0.38,
        description="Regulated category, complex evaluation cycle, high switching cost."
    ),
    AnalogCase(
        name="Analog B — Balanced adoption",
        p=0.008, q=0.42,
        description="Established buyer persona, moderate complexity, clear ROI case."
    ),
    AnalogCase(
        name="Analog C — Rapid, viral adoption",
        p=0.015, q=0.51,
        description="Lower switching costs, strong peer-network signal effects."
    ),
]


# ---------------------------------------------------------------------------
# 3. FITTING — RECOVER (p, q, m) FROM HISTORICAL ADOPTION DATA
# ---------------------------------------------------------------------------

def fit_bass(periods, adopters):
    """
    Fit Bass parameters to observed adoption data using nonlinear least squares.

    Parameters
    ----------
    periods : array-like
        Time index (e.g., [1, 2, 3, ...])
    adopters : array-like
        Observed new adopters per period

    Returns
    -------
    dict with keys p, q, m and confidence intervals
    """
    initial_guess = [0.01, 0.4, np.sum(adopters) * 2]
    bounds = ([1e-5, 1e-3, 1], [0.1, 1.0, np.sum(adopters) * 20])

    popt, pcov = curve_fit(
        bass_adopters, periods, adopters,
        p0=initial_guess, bounds=bounds, maxfev=10000
    )
    perr = np.sqrt(np.diag(pcov))
    return {
        "p": popt[0], "p_se": perr[0],
        "q": popt[1], "q_se": perr[1],
        "m": popt[2], "m_se": perr[2],
    }


# ---------------------------------------------------------------------------
# 4. SYNTHETIC HISTORICAL DATA FOR DEMO
# ---------------------------------------------------------------------------
# Generated from the analog parameters above with noise added — simulates
# what real adoption data for a comparable product might look like.
# ---------------------------------------------------------------------------

def generate_synthetic_history(case, market_size=10000, n_years=10, noise=0.08, seed=42):
    """Create realistic but noisy historical adoption data from an analog case."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n_years + 1)
    true_adopters = bass_adopters(t, case.p, case.q, market_size)
    observed = true_adopters * (1 + rng.normal(0, noise, size=t.shape))
    return t, np.maximum(observed, 0)  # no negative adopters


# ---------------------------------------------------------------------------
# 5. FORECAST: ANCHOR ON ANALOG PARAMETERS, PROJECT FORWARD
# ---------------------------------------------------------------------------

def forecast_new_product(p, q, market_potential, horizon_years=7):
    """Forecast adoption for a new product given p, q, m and a time horizon."""
    t = np.arange(1, horizon_years + 1)
    new_adopters = bass_adopters(t, p, q, market_potential)
    cumulative = bass_cumulative(t, p, q, market_potential)
    return pd.DataFrame({
        "Year": t,
        "New_Adopters": new_adopters.round(0).astype(int),
        "Cumulative_Adopters": cumulative.round(0).astype(int),
        "Penetration_Pct": (cumulative / market_potential * 100).round(1)
    })


# ---------------------------------------------------------------------------
# 6. RUN THE FULL PIPELINE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("BASS DIFFUSION MODEL — DEMO PIPELINE")
    print("=" * 70)

    # --- Step 1: Fit each analog case from synthetic historical data ---
    print("\nStep 1: Fitting Bass parameters to 3 analogous cases\n")
    fitted_params = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, case in zip(axes, ANALOG_CASES):
        t_hist, adopt_hist = generate_synthetic_history(case)
        fit = fit_bass(t_hist, adopt_hist)
        fitted_params.append(fit)

        # Plot observed vs fitted
        t_smooth = np.linspace(0.1, 12, 200)
        ax.bar(t_hist, adopt_hist, alpha=0.35, color="#b8451f",
               edgecolor="#8a3217", label="Observed (synthetic)")
        ax.plot(t_smooth, bass_adopters(t_smooth, fit["p"], fit["q"], fit["m"]),
                color="#181715", linewidth=2.2, label="Bass fit")
        ax.set_title(case.name, fontsize=10, fontweight="bold", color="#181715")
        ax.set_xlabel("Year")
        ax.set_ylabel("New Adopters")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.text(0.02, 0.95,
                f"p = {fit['p']:.4f}\nq = {fit['q']:.3f}\nm = {fit['m']:,.0f}",
                transform=ax.transAxes, fontsize=9, family="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#faf7f1",
                          edgecolor="#d4ccb9"))

        print(f"  {case.name}")
        print(f"    Fitted: p={fit['p']:.4f}, q={fit['q']:.3f}, m={fit['m']:,.0f}")

    fig.suptitle("Bass Model Fit Across Analogous Cases",
                 fontsize=13, fontweight="bold", color="#181715", y=1.02)
    plt.tight_layout()
    plt.savefig("01_analog_fits.png", dpi=160, bbox_inches="tight",
                facecolor="#f3ede2")
    print("\n  → Saved: 01_analog_fits.png")

    # --- Step 2: Triangulate parameters for the new product ---
    # In practice: weight analogs by similarity to target. Here we average.
    p_target = np.mean([f["p"] for f in fitted_params])
    q_target = np.mean([f["q"] for f in fitted_params])

    # Market potential is sized separately (TAM analysis), assumed here.
    market_potential = 50_000

    print(f"\nStep 2: Anchor parameters for new product (averaged across analogs)")
    print(f"  p = {p_target:.4f}  (innovation)")
    print(f"  q = {q_target:.3f}   (imitation)")
    print(f"  m = {market_potential:,}  (assumed market potential from TAM work)")

    # --- Step 3: Forecast 7-year adoption ---
    print(f"\nStep 3: Forecasting 7-year adoption curve\n")
    forecast = forecast_new_product(p_target, q_target, market_potential, 7)
    print(forecast.to_string(index=False))
    forecast.to_csv("02_forecast.csv", index=False)
    print("\n  → Saved: 02_forecast.csv")

    # --- Step 4: Plot the forecast ---
    fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5))
    t_fine = np.linspace(0.1, 8, 200)
    new_fine = bass_adopters(t_fine, p_target, q_target, market_potential)
    cum_fine = bass_cumulative(t_fine, p_target, q_target, market_potential)

    # Sensitivity bands (low / high analog scenarios)
    p_low = min(f["p"] for f in fitted_params)
    q_low = min(f["q"] for f in fitted_params)
    p_high = max(f["p"] for f in fitted_params)
    q_high = max(f["q"] for f in fitted_params)
    low_curve = bass_adopters(t_fine, p_low, q_low, market_potential)
    high_curve = bass_adopters(t_fine, p_high, q_high, market_potential)

    ax2[0].fill_between(t_fine, low_curve, high_curve,
                        color="#b8451f", alpha=0.15,
                        label="Scenario range (low ↔ high analog)")
    ax2[0].plot(t_fine, new_fine, color="#b8451f", linewidth=2.4,
                label="Base case forecast")
    ax2[0].bar(forecast["Year"], forecast["New_Adopters"],
               alpha=0.3, color="#181715", width=0.5)
    ax2[0].set_title("Annual New Adopters", fontsize=12, fontweight="bold",
                     color="#181715")
    ax2[0].set_xlabel("Year since launch")
    ax2[0].set_ylabel("New adopters")
    ax2[0].spines[["top", "right"]].set_visible(False)
    ax2[0].legend(loc="upper right", fontsize=9, frameon=False)

    ax2[1].plot(t_fine, cum_fine, color="#b8451f", linewidth=2.4,
                label="Cumulative adoption")
    ax2[1].axhline(y=market_potential, color="#837e72", linewidth=1,
                   linestyle="--", label=f"Market potential = {market_potential:,}")
    ax2[1].fill_between(t_fine, 0, cum_fine, color="#b8451f", alpha=0.1)
    ax2[1].set_title("Cumulative Adoption", fontsize=12, fontweight="bold",
                     color="#181715")
    ax2[1].set_xlabel("Year since launch")
    ax2[1].set_ylabel("Cumulative adopters")
    ax2[1].spines[["top", "right"]].set_visible(False)
    ax2[1].legend(loc="lower right", fontsize=9, frameon=False)

    fig2.suptitle("7-Year Forecast — New Product (Generic Demo)",
                  fontsize=14, fontweight="bold", color="#181715", y=1.02)
    plt.tight_layout()
    plt.savefig("03_forecast.png", dpi=160, bbox_inches="tight",
                facecolor="#f3ede2")
    print("  → Saved: 03_forecast.png")

    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)
