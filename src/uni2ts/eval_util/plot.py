from typing import Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts import maybe
from gluonts.model import Forecast


def plot_single(
    inp: dict,
    label: dict,
    forecast: Forecast,
    context_length: int,
    intervals: tuple[float, ...] = (0.5, 0.9),
    ax: Optional[plt.axis] = None,
    dim: Optional[int] = None,
    name: Optional[str] = None,
    show_label: bool = False,
    cmap: str = 'viridis' # Add a new parameter for the color map
):
    ax = maybe.unwrap_or_else(ax, plt.gca)
    print('A plotsingle')

    target = np.concatenate([inp["target"], label["target"]], axis=-1)
    start = inp["start"]
    if dim is not None:
        target = target[dim]
        forecast = forecast.copy_dim(dim)
    print('B plotsingle')
    index = pd.period_range(start, periods=len(target), freq=start.freq)
    ax.plot(
        index.to_timestamp()[-context_length - forecast.prediction_length :],
        target[-context_length - forecast.prediction_length :],
        label="target",
        color=cm.get_cmap(cmap)(0.5), # Use the color map for the plot color
    )
    print('C plotsingle')
    forecast.plot(
        intervals=intervals,
        ax=ax,
        color=cm.get_cmap(cmap)(0.75), # Use the color map for the forecast color
        name=name,
        show_label=show_label,
    )
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="lower left")


def plot_next_multi(
    axes: np.ndarray,
    input_it: Iterator[dict],
    label_it: Iterator[dict],
    forecast_it: Iterator[Forecast],
    context_length: int,
    intervals: tuple[float, ...] = (0.5, 0.9),
    dim: Optional[int] = None,
    name: Optional[str] = None,
    show_label: bool = False,
):
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for ax, inp, label, forecast in zip(axes, input_it, label_it, forecast_it):
        plot_single(
            inp,
            label,
            forecast,
            context_length,
            intervals=intervals,
            ax=ax,
            dim=dim,
            name=name,
            show_label=show_label,
        )
