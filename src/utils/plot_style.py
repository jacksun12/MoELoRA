import matplotlib.pyplot as plt


MOPRED_COLORS = ["#DE582B", "#1868B2", "#018A67", "#F3A332", "#8B4513", "#4682B4"]
MOPRED_PATTERNS = ["/", "\\", "x", "||", "--", "++"]


def apply_global_plot_style():
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["legend.framealpha"] = 1.0


def style_axes(ax, grid_axis="y"):
    if grid_axis == "both":
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    else:
        ax.grid(True, axis=grid_axis, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=14, width=1.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def style_legend(ax, loc="upper right", ncol=1, title=None, bbox_to_anchor=None, fontsize=14, title_fontsize=14):
    legend = ax.legend(
        loc=loc,
        ncol=ncol,
        title=title,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
    )
    if legend is not None:
        legend.get_frame().set_linewidth(2.0)
        for text in legend.get_texts():
            text.set_fontweight("bold")
        if legend.get_title() is not None:
            legend.get_title().set_fontweight("bold")
    return legend

