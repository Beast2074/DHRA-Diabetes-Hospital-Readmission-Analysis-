import matplotlib

matplotlib.use("Agg")  # no gui, just save png files
import matplotlib.pyplot as plt

from config import CHART_DIR

# one palette for every chart so the report looks consistent
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"      # main series
ORANGE = "#eb6834"    # comparison series
AQUA = "#1baf7a"      # third series
LIGHT_GRAY = "#d6d5ce"  # de-emphasised bars
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]


def set_style():
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.parse_math": False,  # so "$500" isn't read as latex
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
    })


def subtitle(ax, text):
    # small grey line under the title that says what to look at
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")


def source_note(fig, text="Source: UCI Diabetes 130-US Hospitals (1999-2008), 99,340 eligible encounters"):
    # sits below the x axis label, bbox_inches="tight" keeps it in the png
    fig.text(0.01, -0.03, text, fontsize=8, color=MUTED, ha="left", va="top")


def save(fig, name):
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
