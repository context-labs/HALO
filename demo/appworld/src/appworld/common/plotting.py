from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from typing import Any, cast

from appworld.common.imports import ensure_package_installed
from appworld.common.types import Comparable, Number


def plot_histogram(
    data: list[Comparable | str],
    library: str = "plotext",
    title: str = "",
    bins: int = 100,
    title_size: int | None = None,
    label_size: int | None = None,
    integer_x_labels: bool = False,
    integer_y_labels: bool = False,
    save_path: str | None = None,
    append_to_path: bool = False,  # only for plotext
    use_seaborn: bool = False,  # only for matplotlib
    show_percentages: bool = False,
) -> None:
    ensure_package_installed("numpy")
    import numpy as np

    assert data, "data must be non-empty."
    assert isinstance(data, list), "data must be a list."
    assert library in (
        "plotext",
        "matplotlib",
    ), f"Invalid library: {library}. Choose from plotext or matplotlib."

    if append_to_path and not save_path:
        raise Exception("append_to_path can only be used when save_path is passed.")

    if append_to_path and library != "plotext":
        raise Exception("append_to_path can only be used with plotext library.")

    is_strings = [isinstance(datum, str) for datum in data]
    if any(is_strings) and not all(is_strings):
        raise Exception("Found string histogram. Either all or none must be of string type.")

    if all(is_strings):
        return plot_bar(
            cast(dict[str, Number], Counter(cast(Iterable[str], data))),
            library=library,
            title=title,
            save_path=save_path,
            append_to_path=append_to_path,
        )

    is_datetimes = [isinstance(datum, datetime) for datum in data]
    if not all(is_datetimes) and any(is_datetimes):
        raise Exception("Found datetime histogram. Either all or none must be of datetime type.")

    if any(is_datetimes):
        data_any = cast(list[Any], data)
        min_date, max_date = min(data_any), max(data_any)
        time_stamps = [datum.timestamp() for datum in data_any]
        min_timestamp = min(time_stamps)
        time_stamps = [time_stamp - min_timestamp for time_stamp in time_stamps]
        data = [timestamp / (24 * 60 * 60) for timestamp in time_stamps]  # to days.
        title += f" (in days, from {min_date.to_date_string()} to {max_date.to_date_string()})"

    for datum in data:
        if not isinstance(datum, int | float):
            raise Exception(
                f"Invalid data item: {datum}. All data items must be int or float for plotting histogram."
            )
    data_numeric = cast(list[float | int], data)
    bins = min(bins, len(set(data_numeric)))
    if library == "plotext":
        ensure_package_installed("plotext")
        import plotext as plt

        theme = "clear" if save_path else "pro"
        plt.theme(theme)
        plt.hist(data_numeric, bins)
        if integer_x_labels:
            ticks = plt.xticks()[0]
            rounded_ticks = np.round(ticks).astype(int)  # Round ticks to nearest integer
            plt.xticks(rounded_ticks)  # Set rounded ticks as labels
        if min(data_numeric) != max(data_numeric):
            plt.xlim(min(data_numeric), max(data_numeric))
        plt.title(title)
        plt.plot_size(plt.terminal_width(), 1.5 * plt.terminal_height() / 3)
        if save_path:
            data_string = "\n\n" + plt.build().replace("\x1b[0m", "")
            mode = "a" if append_to_path else "w"
            with open(save_path, mode, encoding="utf-8") as file:
                file.write(data_string)
        else:
            plt.show()
        plt.clf()
    elif library == "matplotlib":
        ensure_package_installed("matplotlib")
        import matplotlib.pyplot as plt

        if use_seaborn:
            ensure_package_installed("seaborn")
            import seaborn as sns

            sns.set_style("white")
            sns.set_style("ticks")
            sns.set_context("talk")
        else:
            plt.style.use("ggplot")
        plt.grid(True)
        if not show_percentages:
            if len(set(data_numeric)) == bins:
                values = sorted(set(data_numeric))
                int_values = list(map(int, values))
                values = list(range(min(int_values), max(int_values) + 1))
                total = len(data_numeric)
                y_s = [round(data_numeric.count(value) / total, 4) for value in values]
                plt.bar(values, y_s, width=0.995, color="orange")
            else:
                plt.hist(data_numeric, bins, color="orange")
        else:
            if len(set(data_numeric)) == bins:
                values = sorted(set(data_numeric))
                int_values = list(map(int, values))
                values = list(range(min(int_values), max(int_values) + 1))
                total = len(data_numeric)
                y_s = [100 * round(data_numeric.count(value) / total, 4) for value in values]
                plt.bar(values, y_s, width=0.995, color="orange")
            else:
                weights = np.ones(len(data_numeric), dtype=float) * (100.0 / len(data_numeric))
                plt.hist(data_numeric, bins, color="orange", weights=weights)
        if integer_x_labels:
            ticks = plt.xticks()[0]
            rounded_ticks = np.round(ticks).astype(int)  # Round ticks to nearest integer
            plt.xticks(rounded_ticks)  # Set rounded ticks as labels
            # plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        if integer_y_labels:
            ticks = plt.yticks()[0]
            rounded_ticks = np.round(ticks).astype(int)
            plt.yticks(rounded_ticks)
        if len(set(data_numeric)) != bins:
            plt.xlim(min(data_numeric), max(data_numeric))
        else:
            plt.xlim(min(data_numeric) - 0.5, max(data_numeric) + 0.5)
        plt.title(title)
        if title_size:
            plt.gca().title.set_size(title_size)
        if label_size:
            plt.tick_params(axis="x", labelsize=label_size)
            plt.tick_params(axis="y", labelsize=label_size)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400)
        else:
            plt.show()
        plt.clf()


def plot_bar(
    data: dict[str, Number],
    library: str = "plotext",
    title: str = "",
    xticks_rotation: bool = False,
    title_size: int | None = None,
    label_size: int | None = None,
    save_path: str | None = None,
    append_to_path: bool = False,  # only for plotext
    use_seaborn: bool = False,  # only for matplotlib
) -> None:
    assert data, "data must be non-empty."
    assert isinstance(data, dict), "data must be a dict."
    assert library in (
        "plotext",
        "matplotlib",
    ), f"Invalid library: {library}. Choose from plotext or matplotlib."

    if append_to_path and not save_path:
        raise Exception("append_to_path can only be used when save_path is passed.")

    if append_to_path and library != "plotext":
        raise Exception("append_to_path can only be used with plotext library.")

    for datum_value in data.values():
        if not isinstance(datum_value, int | float):
            raise Exception(
                f"Invalid data item value: {datum_value}. "
                "All values of data dict must be int/float for bar plots."
            )

    if library == "plotext":
        ensure_package_installed("plotext")
        import plotext as plt

        theme = "clear" if save_path else "pro"
        plt.theme(theme)
        plt.bar(data.keys(), data.values())
        plt.title(title)
        if xticks_rotation:
            raise Exception("xticks_rotation is not supported in plotext.")
        plt.plot_size(plt.terminal_width(), 1.5 * plt.terminal_height() / 3)
        if save_path:
            data_string = "\n\n" + plt.build().replace("\x1b[0m", "")
            mode = "a" if append_to_path else "w"
            with open(save_path, mode, encoding="utf-8") as file:
                file.write(data_string)
        else:
            plt.show()
        plt.clf()
    elif library == "matplotlib":
        ensure_package_installed("matplotlib")
        import matplotlib.pyplot as plt

        if use_seaborn:
            ensure_package_installed("seaborn")
            import seaborn as sns

            sns.set_style("white")
            sns.set_style("ticks")
            sns.set_context("talk")
        else:
            plt.style.use("ggplot")
        plt.grid(True)
        plt.bar(data.keys(), data.values(), color="orange")
        plt.title(title)
        if xticks_rotation:
            plt.xticks(rotation=xticks_rotation)
        plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.0)
        plt.title(title)
        if title_size:
            plt.gca().title.set_size(title_size)
        if label_size:
            plt.tick_params(axis="x", labelsize=label_size)
            plt.tick_params(axis="y", labelsize=label_size)
        if save_path:
            plt.savefig(save_path, dpi=400)
        else:
            plt.show()
        plt.clf()


def plot_line(
    data: dict[str, list[Number]],
    markers: dict[str, str] | None = None,
    library: str = "plotext",
    title: str = "",
    xticks: list[str] | None = None,
    xticks_rotation: bool = False,
    title_size: int | None = None,
    label_size: int | None = None,
    save_path: str | None = None,
    append_to_path: bool = False,  # only for plotext
    use_seaborn: bool = False,  # only for matplotlib
) -> None:
    assert data, "data must be non-empty."
    assert isinstance(data, dict), "data must be a dict."
    assert library in (
        "plotext",
        "matplotlib",
    ), f"Invalid library: {library}. Choose from plotext or matplotlib."

    if append_to_path and not save_path:
        raise Exception("append_to_path can only be used when save_path is passed.")

    if append_to_path and library != "plotext":
        raise Exception("append_to_path can only be used with plotext library.")

    for datum_values in data.values():
        for datum_value in datum_values:
            if not isinstance(datum_value, int | float):
                raise Exception(
                    f"Invalid data item value: {datum_value}. "
                    "All values of data dict must be int/float for line plots."
                )
    if markers is None:
        markers = cast(dict[str, str], dict.fromkeys(data.keys(), "."))
    assert markers is not None  # mypy
    if library == "plotext":
        ensure_package_installed("plotext")
        import plotext as plt

        theme = "clear" if save_path else "pro"
        plt.theme(theme)
        for label, values in data.items():
            plt.plot(values, label=label)
        plt.title(title)
        if xticks_rotation:
            raise Exception("xticks_rotation is not supported in plotext.")
        plt.plot_size(plt.terminal_width(), 1.5 * plt.terminal_height() / 3)
        if save_path:
            data_string = "\n\n" + plt.build().replace("\x1b[0m", "")
            mode = "a" if append_to_path else "w"
            with open(save_path, mode, encoding="utf-8") as file:
                file.write(data_string)
        else:
            plt.show()
        plt.clf()
    elif library == "matplotlib":
        ensure_package_installed("matplotlib")
        import matplotlib.pyplot as plt

        if use_seaborn:
            ensure_package_installed("seaborn")
            import seaborn as sns

            sns.set_style("white")
            sns.set_style("ticks")
            sns.set_context("talk")
        else:
            plt.style.use("ggplot")
            plt.grid(True)
        for label, values in data.items():
            plt.plot(values, label=label, marker=markers[label])
            if xticks:
                plt.xticks(range(len(values)), xticks)
            for xi, yi in enumerate(values):
                plt.annotate(
                    yi,
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                )
        plt.title(title)
        if xticks_rotation:
            plt.xticks(rotation=xticks_rotation)
        plt.tight_layout()
        plt.title(title)
        plt.legend()
        if title_size:
            plt.gca().title.set_size(title_size)
        if label_size:
            plt.tick_params(axis="x", labelsize=label_size)
            plt.tick_params(axis="y", labelsize=label_size)
        if save_path:
            plt.savefig(save_path, dpi=400)
        else:
            plt.show()
        plt.clf()
