"""Figures 2 and 3 of the documentation, from aggregate numbers stated in the text."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent / "figures"
HERE.mkdir(exist_ok=True)

# Figure 2: October's fraud phase on the device-number line, phones per class zone (aggregates).
zones = [
    ("mule", 80),
    ("bot", 29),
    ("chain", 39),
    ("sim_swap", 51),
    ("takeover", 74),
    ("phishing", 43),
    ("qr_swap", 41),
    ("refund", 22),
    ("cashback", 11),
    ("loan", 8),
    ("collusion", 12),
]
fig, ax = plt.subplots(figsize=(10, 2.6))
left = 0
cmap = plt.get_cmap("tab20")
for i, (name, n) in enumerate(zones):
    ax.barh(0, n, left=left, color=cmap(i), edgecolor="white", height=0.6)
    dx = {"cashback": -5, "collusion": 6}.get(name, 0)
    ax.text(
        left + n / 2 + dx,
        0.42 if i % 2 else -0.42,
        f"{name}\n{n}",
        ha="center",
        va="center",
        fontsize=8,
    )
    left += n
ax.barh(0, 25, left=-40, color="#bbbbbb", height=0.6)
ax.text(-27.5, 0, "bot\npool", ha="center", va="center", fontsize=8)
ax.barh(0, 60, left=left + 15, color="#dddddd", height=0.6)
ax.text(left + 45, 0, "legit block\n(dense)", ha="center", va="center", fontsize=8)
ax.set_xlim(-45, left + 80)
ax.set_ylim(-0.9, 0.9)
ax.axis("off")
ax.set_title(
    "October's fraud phase: 410 fresh phones, minted in class order (phones per zone)", fontsize=10
)
fig.tight_layout()
fig.savefig(HERE / "fig2_counter.png", dpi=200)
plt.close(fig)

# Figure 3: the public score path of the uploads.
path = [
    ("five-tree blend", 0.5474),
    ("final-half", 0.5499),
    ("bait-combo", 0.5502),
    ("bait-edge", 0.5506),
    ("research-combo-2", 0.5506),
    ("counter-news", 0.5516),
    ("stack-burst-b8", 0.5531),
    ("stack-burst12-b8", 0.5532),
    ("stack-burst12-sim45", 0.5556),
]
fig, ax = plt.subplots(figsize=(10, 3.6))
xs = range(len(path))
ax.plot(xs, [v for _, v in path], marker="o", color="#1f4e79")
for x, (name, v) in zip(xs, path):
    ax.annotate(
        f"{v:.4f}", (x, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8
    )
ax.set_xticks(list(xs))
ax.set_xticklabels([n for n, _ in path], rotation=25, ha="right", fontsize=8)
ax.set_ylabel("public macro-F1")
ax.set_ylim(0.546, 0.557)
ax.grid(axis="y", alpha=0.3)
ax.set_title("Public score of each upload on the final line", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "fig3_score.png", dpi=200)
plt.close(fig)
print("figures written")

# Figure 1: the final system as a flow (members -> mix -> decoder -> rule layers -> file).
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 8.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, title, body, face="#ffffff", edge="#9aa5b1", bold_edge=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            fc=face,
            ec="#1f4e79" if bold_edge else edge,
            lw=1.6 if bold_edge else 1.0,
        )
    )
    ax.text(x + 1.2, y + h - 2.2, title, fontsize=9.5, fontweight="bold", va="top")
    if body:
        ax.text(x + 1.2, y + h - 5.4, body, fontsize=8, va="top", color="#333333")


def arrow(x, y0, y1):
    ax.annotate(
        "", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>", color="#9aa5b1", lw=1.0)
    )


ax.text(
    50,
    98.5,
    "Evidence and counter feed six members; one mix, a decoder and the rule layers make the file",
    ha="center",
    fontsize=10.5,
    fontweight="bold",
)
box(
    1,
    84,
    98,
    10,
    "Label-free evidence and the device-number counter",
    "flows, bursts, drains, requests, repeats, merchant rings, marks;"
    + chr(10)
    + "the fraud phase segmented by an 11-state HMM, refined with allocation geometry",
)
arrow(50, 84, 81.5)
members = [
    ("Champion", "XGBoost,\ngiven labels"),
    ("LightGBM", "cascade,\ngiven labels"),
    ("CatBoost", "cascade,\ngiven labels"),
    ("Clean-event", "XGBoost,\nevent-cleaned"),
    ("Clean-l2", "XGBoost,\nevent + flips"),
    ("TabM", "neural,\n3 seeds"),
]
for i, (t, b) in enumerate(members):
    x = 1 + i * 16.6
    box(x, 70, 15.0, 11, t, b)
    arrow(x + 7.5, 70, 67.5)
box(
    1,
    57,
    98,
    10,
    "Mix",
    "type split: mean of the five trees;  legit-against-fraud split: TabM 1/2, each tree 1/10",
)
arrow(50, 57, 54.5)
box(
    1,
    45,
    98,
    9,
    "Decoder",
    "expected macro-F1 fixed point, each month on its own; fraud probabilities divided by 0.875",
)
arrow(50, 45, 42.5)
box(
    1,
    33,
    98,
    9,
    "Model rules",
    "look-alike templates take their counter segment or anchor;  operator-role bait rules (phishing_collect)",
)
arrow(50, 33, 30.5)
box(
    1,
    16,
    98,
    14,
    "Base-rate rules (each capped in October rows)",
    "blank-QR merchant payments (10)  ·  mule / bot counter edge (11)  ·  three small repairs (4)\n"
    "counter news: the refined counter's one-class moves (3)  ·  fresh-SIM bursts to velocity_bot (40; cuts set with public reads)",
)
arrow(50, 16, 13.5)
box(
    1,
    2,
    98,
    11,
    "stack-burst12-sim45.csv",
    "260,582 October rows, one label each;  public macro-F1 0.5556;  sha256 c2784900…eb2efc",
    face="#eaf1fb",
    bold_edge=True,
)
fig.savefig(HERE / "fig1_pipeline.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("figure 1 written")
