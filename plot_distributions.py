import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def create_model_comparison_plot():
    means = [10, 20, 30]
    std_dev = 2.0
    x = np.linspace(0, 40, 1000)

    y1 = norm.pdf(x, means[0], std_dev)
    y2 = norm.pdf(x, means[1], std_dev)
    y3 = norm.pdf(x, means[2], std_dev)

    _, ax = plt.subplots(figsize=(12, 7))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    colors = ["#FF6F61", "#6B5B95", "#88B04B"]
    labels = ["Device 1 Data", "Device 2 Data", "Device 3 Data"]

    ax.fill_between(x, y1, color=colors[0], alpha=0.6, label=labels[0])
    ax.fill_between(x, y2, color=colors[1], alpha=0.6, label=labels[1])
    ax.fill_between(x, y3, color=colors[2], alpha=0.6, label=labels[2])

    ax.plot(x, y1, color=colors[0], linewidth=2)
    ax.plot(x, y2, color=colors[1], linewidth=2)
    ax.plot(x, y3, color=colors[2], linewidth=2)

    y_base = -0.02  # Height of line representing smaller model

    for i, m in enumerate(means):
        ax.plot([m - 3, m + 3], [y_base, y_base], color=colors[i], linewidth=4)
        ax.text(
            m,
            y_base - 0.015,
            f"On-device ML Model {i+1}\n(Specialized)",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=colors[i],
        )
        ax.vlines(m, 0, y_base, linestyles="dotted", colors=colors[i], alpha=0.5)

    y_large = y_base - 0.08  # Height of line representing large model

    ax.plot(
        [means[0] - 4, means[2] + 4], [y_large, y_large], color="#333333", linewidth=6
    )
    ax.text(
        means[1],
        y_large - 0.015,
        "LARGE MASTER ML MODEL\n(Generalist)",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#333333",
    )
    ax.annotate(
        "",
        xy=(means[0], y_large + 0.005),
        xytext=(means[0], y_base - 0.04),
        arrowprops=dict(arrowstyle="<->", color="gray", lw=1),
    )
    ax.annotate(
        "",
        xy=(means[1], y_large + 0.005),
        xytext=(means[1], y_base - 0.04),
        arrowprops=dict(arrowstyle="<->", color="gray", lw=1),
    )
    ax.annotate(
        "",
        xy=(means[2], y_large + 0.005),
        xytext=(means[2], y_base - 0.04),
        arrowprops=dict(arrowstyle="<->", color="gray", lw=1),
    )
    ax.set_title("Model Capacity vs. Distribution Coverage", fontsize=16, pad=20)
    ax.set_ylim(-0.15, 0.25)
    ax.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    create_model_comparison_plot()
