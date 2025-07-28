import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal
from cycler import cycler
from qualang_tools.plot.fitting import Fit
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

def plot_resonator_spectroscopy(npz_path, resonator_LO):
    """
    Load a .npz containing 'IF_frequencies', 'I_data', 'Q_data',
    compute R & phase, plot in Qiskit style with extra spacing, then fit.
    """

    data = np.load(npz_path)
    freqs_Hz = data['IF_frequencies']
    freqs_MHz = freqs_Hz / 1e6
    I, Q = data['I_data'], data['Q_data']
    S = I + 1j * Q
    R = np.abs(S)
    phase = np.unwrap(np.angle(S))
    phase = signal.detrend(phase)

    mpl.rcParams.update({
        'figure.figsize':     (8, 6),
        'font.size':           12,
        'axes.titlesize':      14,
        'axes.labelsize':      12,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.grid':           True,
        'grid.linestyle':      '--',
        'grid.color':         '0.8',
        'xtick.direction':    'in',
        'ytick.direction':    'in',
        'xtick.minor.visible':True,
        'ytick.minor.visible':True,
        'lines.markersize':    5,
    })

    # --- 3) make the 2‑panel plot with extra vertical spacing ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={'hspace': 0.4}  # increase space between panels
    )
    ax1.plot(freqs_MHz, R, '.', markeredgewidth=0.5)
    ax1.set_ylabel(r'$R = \sqrt{I^2 + Q^2}$ [V]')
    ax2.plot(freqs_MHz, phase, '.', markeredgewidth=0.5)
    ax2.set_xlabel('Intermediate frequency [MHz]')
    ax2.set_ylabel('Phase [rad]')
    fig.suptitle(f"Resonator spectroscopy — LO = {resonator_LO:.3f} GHz")
    plt.tight_layout(rect=[0,0,1,0.95])

    # --- fit the data but don’t auto-plot ---
    fit = Fit()
    res_spec_fit = fit.reflection_resonator_spectroscopy(
        freqs_Hz / u.MHz,
        R,
        plot=False
    )
    # x‑axis for plotting is just your original IF freq in MHz
    x_fit = freqs_Hz / 1e6  

    # y_fit from the fit_func; it expects the same units you passed in
    y_fit = res_spec_fit["fit_func"](freqs_Hz/u.MHz)

    # now plot on ax1
    ax1.plot(
        x_fit,
        y_fit,
        'g-',
        linewidth=2,
        label=f"f = {res_spec_fit['f'][0]:.3f} ± {res_spec_fit['f'][1]:.3f} MHz"
    )
    ax1.legend(loc='best')

    # ensure ax1 has the right labels/title
    ax1.set_ylabel(r"$R = \sqrt{I^2 + Q^2}$ [V]")

    # leave ax2 for phase
    ax2.set_ylabel("Phase [rad]")
    plt.show()

def plot_qubit_spectroscopy(npz_path: str):
    """
    Load a spectroscopy .npz file with keys:
      - 'frequencies' : array of detunings in Hz
      - 'I_data'      : I quadrature [V]
      - 'Q_data'      : Q quadrature [V]
    and plot amplitude & phase in a Qiskit‑inspired Matplotlib style,
    plus a dashed red Lorentzian fit on the amplitude.
    """
    # 1) Load
    data     = np.load(npz_path)
    freqs_hz = data["frequencies"]
    I        = data["I_data"]
    Q        = data["Q_data"]
    
    # 2) Compute metrics
    R   = np.sqrt(I**2 + Q**2)
    phi = np.angle(I + 1j * Q)
    
    # 3) Convert to MHz
    freqs_mhz = freqs_hz / 1e6
    
    # 4) Qiskit‑like style
    qiskit_style = {
        # figure & text
        "figure.figsize":       (6, 4),
        "font.family":          "sans-serif",
        "font.size":            12,
        "axes.titlesize":       14,
        "axes.labelsize":       12,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        # spines & ticks
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.top":            False,
        "ytick.right":          False,
        # grid
        "axes.grid":            True,
        "grid.linestyle":       "--",
        "grid.linewidth":       0.8,
        "grid.color":           "#cccccc",
        # markers
        "lines.marker":         "o",
        "lines.markersize":     4,
        "lines.linestyle":      "None",
        "lines.markeredgewidth": 0.0,
        "lines.markerfacecolor": "#1f77b4",   # fill color of the circles
        "lines.markeredgewidth": 0.8,         # outline thickness
    }
    plt.rcParams.update(qiskit_style)
    
    # 5) Fit a Lorentzian to R_vs_freq
    def lorentzian(x, A, center, width, offset):
        return offset + A / (1 + ((x - center) / width) ** 2)
    
    # initial guesses
    A0      = R.max() - R.min()
    center0 = freqs_mhz[np.argmax(R)]
    width0  = (freqs_mhz.max() - freqs_mhz.min()) / 10
    offset0 = R.min()
    p0 = [A0, center0, width0, offset0]
    
    popt, pcov = curve_fit(lorentzian, freqs_mhz, R, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    
    # build a smooth fit curve
    x_fit = np.linspace(freqs_mhz.min(), freqs_mhz.max(), 500)
    y_fit = lorentzian(x_fit, *popt)
    
    # 6) Plot
    fig, (axR, axP) = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Qubit Spectroscopy", y=1.02)
    lable_fs = 10
    # amplitude + fit
    axR.plot(freqs_mhz, R, color="#1f77b4", label="Data", ms=5)
    axR.plot(x_fit,    y_fit, color="red", linestyle="--", alpha=0.4,
             label=f"Fit: ν₀={popt[1]:.3f}±{perr[1]:.3f} MHz")
    axR.set_ylabel(r"Amplitude $R=\sqrt{I^2+Q^2}$ [V]", fontsize=lable_fs)
    axR.legend(loc="best", fontsize=10)
    
    # phase
    axP.plot(freqs_mhz, phi, color="#1f77b4", label="Phase")
    axP.set_xlabel("Detuning [MHz]")
    axP.set_ylabel("Phase [rad]", fontsize=lable_fs)
    
    plt.tight_layout()
    plt.show()

def plot_t1(npz_path):
    """
    Load a .npz with keys ['taus','I_data','Q_data'] and
    plot a Qiskit‑like T1 decay curve using pure Matplotlib styling.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file.
    """
    # ——— 1) load data ——————————————————————————————
    data = np.load(npz_path)
    taus = data['taus']       # shape (N,), in units where 4*taus = delay in ns
    I = data['I_data']
    Q = data['Q_data']

    # ——— 2) build axes & projection ——————————————
    delays_ns = 4 * taus
    delays_us = delays_ns * 1e-3            # convert to µs for x‑axis

    # approximate “main‑axis” projection by magnitude
    proj = np.sqrt(I**2 + Q**2)
    proj_norm = (proj - proj.min()) / (proj.max() - proj.min())

    # ——— 3) tweak Matplotlib defaults —————————————
    plt.rcParams.update({
        # grid
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.color':   '#D3D3D3',
        'grid.linewidth': 0.8,
        # spines
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        # ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        # lines & markers
        'lines.linewidth': 1.6,
        'lines.marker':    'o',
        'lines.markersize': 6,
        # font
        'font.size': 10,
    })

    # ——— 4) scatter the data ———————————————————————
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(delays_us, proj_norm,
               color='#1077C1',        # Qiskit’s primary blue
               alpha=0.85,
               label='Data')

    # ——— 5) run the same T1 fit —————————————————————
    fit = Fit()
    fit.T1(delays_ns, proj_norm, plot=True)

    # ——— 6) finalize labels & legend —————————————
    fs = 11
    ax.set_title('T1 measurement', pad=10, fontsize=fs)
    ax.set_xlabel('Delay [µs]', fontsize=fs)
    ax.set_ylabel('P(0)', fontsize=fs)
    ax.legend(loc='best', frameon=True, fontsize=fs)

    plt.tight_layout()
    plt.show()

def plot_ramsey(npz_path,
                
                cut_detuning_idx: int = None,
                convert_to_probability: bool = False,
                perform_fit: bool = False):
    """
    Load a Ramsey .npz and plot it in a Qiskit‑like style (manual rcParams).

    - 2D I_data → chevron heatmaps + optional 1D cut by index.
    - 1D I_data → simple 1D trace.
    
    Parameters
    ----------
    npz_path : str
        Path to the .npz containing at least 'taus' and 'I_data'.
        Optional keys: 'IF_frequencies', 'Q_data'.
    cut_detuning_idx : int, optional
        If I_data is 2D, index into the frequency axis to extract
        the I‐vs‐delay curve.
    convert_to_probability : bool
        If True, attempts I → P(1) mapping.  Not implemented
        without calibration matrix.
    perform_fit : bool
        If True, runs Fit().ramsey(...) on the 1D trace.
    """
    # 1) Qiskit‑like styling via rcParams
    plt.rcParams.update({
        'figure.facecolor':'white','axes.facecolor':'white',
        'axes.edgecolor':'#444','axes.labelsize':12,
        'axes.titlesize':14,'axes.titleweight':'bold',
        'axes.grid':True,'grid.color':'#e5e5e5','grid.linestyle':'-',
        'xtick.direction':'out','ytick.direction':'out',
        'xtick.color':'#444','ytick.color':'#444',
        'xtick.labelsize':10,'ytick.labelsize':10,
        'lines.linewidth':1.5,'lines.markersize':6,
        'legend.frameon':False,'image.cmap':'viridis',
    })
    plt.rcParams['axes.prop_cycle'] = cycler('color',[
        '#006BB2','#FF9F1C','#0EAD69','#992E5E','#4AB05D',
    ])

    # 2) Load data
    d      = np.load(npz_path)
    taus   = d['taus']
    idle_ns= 4 * taus
    I      = d['I_data']
    Q      = d.get('Q_data', None)
    freqs  = d.get('IF_frequencies', None)

    # 3) Branch on dimension
    if I.ndim == 2 and freqs is not None:
        detunings = freqs / 1e6  # in MHz

        # 3a) Chevron heatmaps
        fig, (axI, axQ) = plt.subplots(2,1, figsize=(8,6), constrained_layout=True)
        # position the suptitle via y instead of pad
        fig.suptitle('Ramsey Chevron', y=0.96)

        axI.pcolormesh(detunings, idle_ns, I, shading='auto')
        axI.set(title='I quadrature [V]', ylabel='Idle time [ns]')

        if Q is not None and Q.ndim == 2:
            axQ.pcolormesh(detunings, idle_ns, Q, shading='auto')
            axQ.set(title='Q quadrature [V]',
                   xlabel='Qubit detuning [MHz]', ylabel='Idle time [ns]')
        else:
            axQ.remove()

        # 3b) Optional 1D cut by index
        if cut_detuning_idx is not None:
            if not (0 <= cut_detuning_idx < detunings.size):
                raise IndexError(f"cut_detuning_idx must be in [0, {detunings.size-1}]")
            det0 = detunings[cut_detuning_idx]
            yI   = I[:, cut_detuning_idx]
            delay_us = idle_ns / 1e3

            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.plot(delay_us, yI, 'o-', label='I quadrature')

            if convert_to_probability:
                raise NotImplementedError(
                    "I → P(1) conversion requires a discrimination calibration matrix."
                )
            else:
                ax2.set_ylabel('I quadrature [V]')

            ax2.set(xlabel='Delay [μs]',
                    title=f'Cut idx={cut_detuning_idx} → {det0:.1f} MHz')
            ax2.grid(True); ax2.legend(loc='upper right')

            if perform_fit:
                from qualang_tools.plot.fitting import Fit
                fit = Fit()
                fit.ramsey(idle_ns, yI, plot=True)
                ax2.set(xlabel='Idle time [ns]',
                       ylabel='I quadrature [V]',
                       title='Ramsey measurement with detuned gates')

    elif I.ndim == 1:
        # 1D‐only trace
        delay_us = idle_ns / 1e3
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(delay_us, I, 'o-', label='I quadrature')

        if convert_to_probability:
            raise NotImplementedError(
                "I → P(1) conversion requires a discrimination calibration matrix."
            )
        else:
            ax.set_ylabel('I quadrature [V]')

        ax.set(xlabel='Delay [μs]', title='Ramsey trace (1D data)')
        ax.grid(True); ax.legend(loc='upper right')

        if perform_fit:
            from qualang_tools.plot.fitting import Fit
            fit = Fit()
            fit.ramsey(idle_ns, I, plot=True)
            ax.set(xlabel='Idle time [ns]',
                   ylabel='I quadrature [V]',
                   title='Ramsey measurement')

    else:
        raise ValueError(f"Unsupported I_data ndim={I.ndim}")

    plt.show()


def plot_rb(npz_path, delta_clifford=10, figsize=(6,6), bottom=0.15):
    """
    Load I_data/Q_data from the given .npz, average over sequences,
    fit to A*p**x + B, print the fit summary, and draw it at the bottom
    of the figure.
    """
    # 1) load & average ----------------------------------------------------------------
    data    = np.load(npz_path)
    I       = data["I_data"]               # (num_sequences, num_lengths)
    mean_I  = np.mean(I, axis=0)
    std_I   = np.std(I,  axis=0)

    # 2) build x‑axis -------------------------------------------------------------------
    n_points       = mean_I.size
    x = np.arange(1, n_points+1) * delta_clifford

    # 3) fit to power law ---------------------------------------------------------------
    def power_law(x, A, B, p):
        return A * p**x + B

    p0    = [0.5, 0.5, 0.9]
    pars, cov = curve_fit(power_law, x, mean_I, p0=p0, maxfev=2000)
    stdevs   = np.sqrt(np.diag(cov))

    # 4) compute error‑rates ------------------------------------------------------------
    one_minus_p = 1 - pars[2]
    r_c       = one_minus_p * (1 - 1/2**1)
    r_c_s     = stdevs[2]   * (1 - 1/2**1)
    r_g       = r_c / 1.875
    r_g_s     = r_c_s / 1.875

    # 5) print to console ----------------------------------------------------------------
    print("#########################")
    print("### Fitted Parameters ###")
    print("#########################")
    print(f"A = {pars[0]:.3} ({stdevs[0]:.1}), "
          f"B = {pars[1]:.3} ({stdevs[1]:.1}), "
          f"p = {pars[2]:.3} ({stdevs[2]:.1})")
    print("Covariance Matrix")
    print(cov)
    print("#########################")
    print("### Useful Parameters ###")
    print("#########################")
    print(
        f"Error rate: 1-p = {np.format_float_scientific(one_minus_p, precision=2)}"
        f" ({stdevs[2]:.1})\n"
        f"Clifford set infidelity: r_c = "
        f"{np.format_float_scientific(r_c, precision=2)} ({r_c_s:.1})\n"
        f"Gate infidelity: r_g = "
        f"{np.format_float_scientific(r_g, precision=2)} ({r_g_s:.1})"
    )

    # 6) build the same multi-line block for the figure -------------------------------
    status_text = (
        "### Fitted Parameters ###\n"
        f"A = {pars[0]:.3} ({stdevs[0]:.1}), "
        f"B = {pars[1]:.3} ({stdevs[1]:.1}), "
        f"p = {pars[2]:.3} ({stdevs[2]:.1})\n"
        "Covariance Matrix:\n"
        f"{cov[0]}\n{cov[1]}\n{cov[2]}\n"
        "### Useful Parameters ###\n"
        f"1-p = {one_minus_p:.2e} ({stdevs[2]:.1})  "
        f"r_c = {r_c:.2e} ({r_c_s:.1})  "
        f"r_g = {r_g:.2e} ({r_g_s:.1})"
    )

    # 7) plot ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    # move the axes up to make room for the status box
    fig.subplots_adjust(bottom=bottom)

    # errorbars
    ax.errorbar(
        x, mean_I,
        yerr=std_I,
        fmt='o',
        color='gray',
        ecolor='lightgray',
        capsize=3,
        alpha=0.8,
    )
    # fit curve
    x_fit = np.linspace(x.min(), x.max(), 200)
    ax.plot(
        x_fit,
        power_law(x_fit, *pars),
        linestyle='-',
        linewidth=2,
        color='tab:blue',
    )

    ax.set_xlabel("Clifford Length")
    ax.set_ylabel("P(0)")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlim(0, x.max() + delta_clifford*0.1)

    # 8) draw status box at bottom -----------------------------------------------------
    fig.text(
        0.5,        # horizontally centered
        0.02,       # 2% up from the bottom of the figure
        status_text,
        ha='center',
        va='bottom',
        fontsize=10,
        family='monospace',
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="black")
    )

    plt.tight_layout(rect=[0, 0.30, 1, 1])
    plt.show()