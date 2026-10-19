#!/usr/bin/env python3
"""
nanopulldown.report.theme
=====================
One place for the matplotlib report figures' look: surface, ink, rules and
the categorical / sequential palettes, in a light and a dark variant.
`report.figure_theme` (config.yaml) picks it -- "dark" matches the report's
dark stylesheet (report/custom.css); `report.figure_transparent: true` saves
with no background at all, so the figure sits directly on the report page.

Palettes are the dataviz skill's reference instance: the same eight hues,
stepped per mode (the dark column was chosen for a dark surface, >= 3:1
contrast). Roles, not raw hex, are what the plot scripts use:

    th = theme()            # from config
    th.apply()              # matplotlib rcParams
    th.color("domain")      # role -> hex
    th.cmap()               # sequential map, "near zero" recedes into the surface
    th.cell_ink(v)          # annotation colour on a cmap cell of value v in [0, 1]
    th.save(fig, path)      # honours figure_transparent
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from nanopulldown.core.config import load_config

# categorical slots (reference palette): light / dark steps of the same hue
_SLOTS = {
    "blue": ("#2a78d6", "#3987e5"), "orange": ("#eb6834", "#d95926"),
    "aqua": ("#1baf7a", "#199e70"), "yellow": ("#eda100", "#c98500"),
    "magenta": ("#e87ba4", "#d55181"), "green": ("#008300", "#008300"),
    "violet": ("#6250d6", "#9085e9"), "red": ("#e34948", "#e66767"),
    "gray": ("#9a9893", "#7d8a99"),
}
# semantic role -> slot
_ROLES = {
    "family": "violet", "domain": "blue", "homologous_superfamily": "aqua", "repeat": "orange",
    "site": "magenta", "unintegrated": "gray",
    "disorder": "red", "binding": "green", "linker_track": "yellow",
    "coils": "yellow", "signal_peptide": "violet", "tm": "green",
}
# series order for line plots (fixed, never cycled past 8)
_ORDER = ["blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red"]

_MODES = {
    "light": {"surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "rule": "#d9d8d4",
              "seq": ["#ffffff", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]},
    # surface = the report's panel colour (report/custom.css --ab-panel)
    "dark": {"surface": "#172230", "ink": "#e8eef5", "ink2": "#93a4b6", "rule": "#2c3e52",
             "seq": ["#172230", "#1c3352", "#184f95", "#2a78d6", "#6da7ec", "#cde2fb"]},
}


def _mix(c1: str, c2: str, a: float) -> str:
    """a * c1 + (1 - a) * c2 (sRGB hex)."""
    x = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    y = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a * p + (1 - a) * q):02x}" for p, q in zip(x, y))


@dataclass
class Theme:
    mode: str = "dark"
    transparent: bool = False
    m: dict = field(init=False)

    def __post_init__(self):
        self.mode = self.mode if self.mode in _MODES else "dark"
        self.m = _MODES[self.mode]

    # ── colours ───────────────────────────────────────────────────────────
    @property
    def dark(self) -> bool:
        return self.mode == "dark"

    def __getattr__(self, k):                       # th.surface, th.ink, th.ink2, th.rule
        if k in ("surface", "ink", "ink2", "rule"):
            return self.m[k]
        raise AttributeError(k)

    def slot(self, name: str) -> str:
        return _SLOTS[name][1 if self.dark else 0]

    def color(self, role: str) -> str:
        return self.slot(_ROLES.get(role, role))

    def series(self) -> list[str]:
        return [self.slot(s) for s in _ORDER]

    def region(self, kind: str) -> str:
        """Domain-map segment fills: domain = full blue; disordered / linker are
        quiet tints (light) or shades (dark) so domains stay the figure."""
        if kind == "domain":
            return self.color("domain")
        base = self.slot("orange") if kind == "disordered" else self.slot("gray")
        return _mix(base, self.surface, 0.45 if self.dark else 0.40)

    def cmap(self):
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list(f"seq_{self.mode}", self.m["seq"])

    def cell_ink(self, v: float) -> str:
        """Readable annotation on a cmap cell: light map gets dark ink low / white
        high; dark map (bright at high values) the other way round."""
        if self.dark:
            return "#0b1118" if v >= 0.75 else self.ink
        return "white" if v >= 0.55 else "#1f2937"

    # ── matplotlib ────────────────────────────────────────────────────────
    def apply(self) -> None:
        import matplotlib as mpl
        from cycler import cycler
        bg = "none" if self.transparent else self.surface
        mpl.rcParams.update({
            "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
            "savefig.edgecolor": bg, "text.color": self.ink, "axes.labelcolor": self.ink2,
            "axes.titlecolor": self.ink, "axes.edgecolor": self.ink2,
            "xtick.color": self.ink2, "ytick.color": self.ink2,
            "xtick.labelcolor": self.ink2, "ytick.labelcolor": self.ink2,
            "legend.labelcolor": self.ink, "legend.facecolor": bg, "legend.edgecolor": self.rule,
            "grid.color": self.rule, "axes.prop_cycle": cycler(color=self.series()),
            "svg.fonttype": "path",
        })

    def save(self, fig, path, **kw) -> None:
        fig.savefig(path, transparent=self.transparent, **kw)

    def save_all(self, fig, out, formats: str, png_dpi: int | None = None, **kw) -> None:
        """Save `fig` once per comma-joined format, next to `out` (its suffix swapped)."""
        stem = Path(out).with_suffix("")
        for fmt in formats.split(","):
            fmt = fmt.strip()
            extra = {"dpi": png_dpi} if png_dpi and fmt == "png" else {}
            self.save(fig, f"{stem}.{fmt}", **extra, **kw)


def theme(cfg: dict | None = None) -> Theme:
    rpt = (cfg or load_config()).get("report") or {}
    return Theme(str(rpt.get("figure_theme", "dark")), bool(rpt.get("figure_transparent", False)))


class _LazyTheme:
    """theme() built on first attribute access, so a figure module can keep a
    module-level TH without reading the config at import."""

    def __init__(self, cfg: dict | None = None):
        self._cfg, self._th = cfg, None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._th is None:
            self._th = theme(self._cfg)
        return getattr(self._th, name)


def lazy_theme(cfg: dict | None = None) -> _LazyTheme:
    return _LazyTheme(cfg)
