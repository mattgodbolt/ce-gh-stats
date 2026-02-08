from pathlib import Path
from collections import defaultdict
import click
import json
import datetime
from github import Github
import os
from dataclasses import dataclass
import typing


@dataclass
class Context:
    access_token: str
    _github: typing.Optional[Github] = None

    @property
    def github(self) -> Github:
        if self._github is None:
            self._github = Github(self.access_token)
        return self._github


@click.group()
@click.pass_context
@click.option(
    "--access-token", default=os.getenv("GITHUB_TOKEN"), help="token to use (defaults to env var GITHUB_TOKEN)"
)
def cli(ctx, access_token: str):
    ctx.obj = Context(access_token=access_token)


@cli.command()
@click.option("--organization", default="compiler-explorer")
@click.option("--project", default="compiler-explorer")
@click.argument("output", type=click.File(mode="a", encoding="utf8"))
@click.pass_obj
def stats(ctx: Context, output: typing.TextIO, organization: str, project: str) -> None:
    """Append JSON information to OUTPUT."""
    org = ctx.github.get_organization(organization)
    repo = org.get_repo(project)
    open = defaultdict(int)
    closed = defaultdict(int)
    with click.progressbar(repo.get_issues(state="all"), label="Reading issues") as issue_list:
        for issue in issue_list:
            to_update = open if issue.state == "open" else closed
            for label in issue.labels:
                to_update[label.name] += 1
    head_revision = repo.get_branch(repo.default_branch).commit.sha
    result = dict(
        as_of=datetime.datetime.now().isoformat(),
        head_revision=head_revision,
        languages=repo.get_languages(),
        issues=dict(open=open, closed=closed),
        open_issues_count=repo.open_issues_count,
        watchers_count=repo.watchers_count,
        stargazers_count=repo.stargazers_count,
        forks_count=repo.forks_count,
    )
    json.dump(result, output)
    output.write("\n")


# -- Colour constants --
CE_GREEN = "#67c52a"
CE_DARK = "#282828"
CE_LIGHT_GREY = "#aaaaaa"
CE_ORANGE = "#e8a317"
CE_BLUE = "#3b8eed"
CE_RED = "#e84040"
CE_PURPLE = "#b467e8"


def load_stats(path: str) -> list[dict]:
    """Read NDJSON stats file, returning records that have at least an as_of field."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "as_of" in record:
                records.append(record)
    return records


def ce_style(fig, ax) -> None:
    """Apply the Compiler Explorer dark theme to a figure and axes."""
    fig.set_facecolor(CE_DARK)
    ax.set_facecolor(CE_DARK)
    ax.tick_params(colors=CE_LIGHT_GREY)
    ax.grid(True, color=CE_LIGHT_GREY, alpha=0.2)
    for spine in ax.spines.values():
        spine.set_color(CE_LIGHT_GREY)
        spine.set_alpha(0.3)


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=150, facecolor=CE_DARK)
    import matplotlib.pyplot as plt

    plt.close(fig)
    click.echo(f"Saved {out / name}")


def plot_open_issues(records: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    dates = []
    counts = []
    for r in records:
        if "open_issues_count" not in r:
            continue
        dates.append(datetime.datetime.fromisoformat(r["as_of"]))
        counts.append(r["open_issues_count"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ce_style(fig, ax)
    ax.plot(dates, counts, linewidth=1.5, color=CE_GREEN)
    ax.set_title("Compiler Explorer — Open Issues Over Time", color="white", fontsize=14)
    ax.set_xlabel("Date", color=CE_LIGHT_GREY)
    ax.set_ylabel("Open Issues", color=CE_LIGHT_GREY)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, out, "open-issues.png")


def plot_stars_and_forks(records: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    dates, stars, forks = [], [], []
    for r in records:
        if "stargazers_count" not in r:
            continue
        dates.append(datetime.datetime.fromisoformat(r["as_of"]))
        stars.append(r["stargazers_count"])
        forks.append(r["forks_count"])

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ce_style(fig, ax1)
    ax1.plot(dates, stars, linewidth=1.5, color=CE_GREEN, label="Stars")
    ax1.set_ylabel("Stars", color=CE_GREEN)
    ax1.tick_params(axis="y", labelcolor=CE_GREEN)

    ax2 = ax1.twinx()
    ax2.plot(dates, forks, linewidth=1.5, color=CE_ORANGE, label="Forks")
    ax2.set_ylabel("Forks", color=CE_ORANGE)
    ax2.tick_params(axis="y", labelcolor=CE_ORANGE)
    ax2.tick_params(axis="x", colors=CE_LIGHT_GREY)
    for spine in ax2.spines.values():
        spine.set_color(CE_LIGHT_GREY)
        spine.set_alpha(0.3)

    ax1.set_title("Compiler Explorer — Stars & Forks", color="white", fontsize=14)
    ax1.set_xlabel("Date", color=CE_LIGHT_GREY)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        facecolor=CE_DARK,
        edgecolor=CE_LIGHT_GREY,
        labelcolor="white",
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, out, "stars-and-forks.png")


def plot_languages(records: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Subsample to every 3rd record (~1/day) for a cleaner stacked area
    sampled = [r for i, r in enumerate(records) if i % 3 == 0 and "languages" in r]

    dates = []
    ts_pct, js_pct, py_pct, other_pct = [], [], [], []
    for r in sampled:
        langs = r["languages"]
        total = sum(langs.values())
        if total == 0:
            continue
        dates.append(datetime.datetime.fromisoformat(r["as_of"]))
        ts_pct.append(langs.get("TypeScript", 0) / total * 100)
        js_pct.append(langs.get("JavaScript", 0) / total * 100)
        py_pct.append(langs.get("Python", 0) / total * 100)
        other_pct.append(100 - ts_pct[-1] - js_pct[-1] - py_pct[-1])

    fig, ax = plt.subplots(figsize=(12, 5))
    ce_style(fig, ax)
    ax.stackplot(
        dates,
        ts_pct,
        js_pct,
        py_pct,
        other_pct,
        labels=["TypeScript", "JavaScript", "Python", "Other"],
        colors=[CE_GREEN, CE_ORANGE, CE_BLUE, "#666666"],
        alpha=0.85,
    )
    ax.set_ylim(0, 100)
    ax.set_title("Compiler Explorer — Language Composition", color="white", fontsize=14)
    ax.set_xlabel("Date", color=CE_LIGHT_GREY)
    ax.set_ylabel("% of codebase (by bytes)", color=CE_LIGHT_GREY)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.legend(loc="center right", facecolor=CE_DARK, edgecolor=CE_LIGHT_GREY, labelcolor="white")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, out, "languages.png")


def plot_issue_categories(records: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    labels_to_track = {
        "bug": CE_RED,
        "request": CE_GREEN,
        "enhancement": CE_BLUE,
        "new-compilers": CE_ORANGE,
        "new-libs": CE_PURPLE,
    }

    dates = []
    series = {label: [] for label in labels_to_track}
    for r in records:
        if "issues" not in r:
            continue
        open_issues = r["issues"].get("open", {})
        dates.append(datetime.datetime.fromisoformat(r["as_of"]))
        for label in labels_to_track:
            series[label].append(open_issues.get(label, 0))

    fig, ax = plt.subplots(figsize=(12, 5))
    ce_style(fig, ax)
    for label, colour in labels_to_track.items():
        ax.plot(dates, series[label], linewidth=1.5, color=colour, label=label)
    ax.set_title("Compiler Explorer — Open Issues by Category", color="white", fontsize=14)
    ax.set_xlabel("Date", color=CE_LIGHT_GREY)
    ax.set_ylabel("Open Issues", color=CE_LIGHT_GREY)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.legend(loc="upper left", facecolor=CE_DARK, edgecolor=CE_LIGHT_GREY, labelcolor="white")
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, out, "issue-categories.png")


def plot_extrapolations(records: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    def to_days(dt_list):
        """Convert datetimes to float days since epoch for polyfit."""
        epoch = datetime.datetime(2020, 1, 1)
        return np.array([(d - epoch).total_seconds() / 86400 for d in dt_list])

    def from_days(day_val):
        epoch = datetime.datetime(2020, 1, 1)
        return epoch + datetime.timedelta(days=float(day_val))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.set_facecolor(CE_DARK)
    fig.suptitle("Compiler Explorer — Fun Extrapolations", color="white", fontsize=16, y=0.98)

    for ax in axes.flat:
        ax.set_facecolor(CE_DARK)
        ax.tick_params(colors=CE_LIGHT_GREY, labelsize=8)
        ax.grid(True, color=CE_LIGHT_GREY, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color(CE_LIGHT_GREY)
            spine.set_alpha(0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    # -- Panel 1: 20k Stars --
    ax = axes[0, 0]
    dates, vals = [], []
    for r in records:
        if "stargazers_count" in r:
            dates.append(datetime.datetime.fromisoformat(r["as_of"]))
            vals.append(r["stargazers_count"])
    x = to_days(dates)
    coeffs = np.polyfit(x, vals, 1)
    target = 20000
    target_day = (target - coeffs[1]) / coeffs[0]
    target_date = from_days(target_day)

    future_end = to_days([target_date + datetime.timedelta(days=90)])[0]
    trend_x = np.linspace(x[0], future_end, 200)
    trend_y = np.polyval(coeffs, trend_x)
    trend_dates = [from_days(d) for d in trend_x]

    ax.plot(dates, vals, linewidth=1.5, color=CE_GREEN)
    ax.plot(trend_dates, trend_y, "--", linewidth=1, color=CE_LIGHT_GREY, alpha=0.7)
    ax.axhline(y=target, color=CE_ORANGE, linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title("20,000 Stars", color="white", fontsize=11)
    ax.set_ylabel("Stars", color=CE_LIGHT_GREY, fontsize=9)
    ax.annotate(
        f"~{target_date.strftime('%b %Y')}",
        xy=(target_date, target),
        color=CE_ORANGE,
        fontsize=9,
        ha="center",
        va="bottom",
    )

    # -- Panel 2: 1,000 Open Issues --
    ax = axes[0, 1]
    dates, vals = [], []
    for r in records:
        if "open_issues_count" in r:
            dates.append(datetime.datetime.fromisoformat(r["as_of"]))
            vals.append(r["open_issues_count"])
    x = to_days(dates)
    coeffs = np.polyfit(x, vals, 1)
    target = 1000
    target_day = (target - coeffs[1]) / coeffs[0]
    target_date = from_days(target_day)

    future_end = to_days([target_date + datetime.timedelta(days=90)])[0]
    trend_x = np.linspace(x[0], future_end, 200)
    trend_y = np.polyval(coeffs, trend_x)
    trend_dates = [from_days(d) for d in trend_x]

    ax.plot(dates, vals, linewidth=1.5, color=CE_GREEN)
    ax.plot(trend_dates, trend_y, "--", linewidth=1, color=CE_LIGHT_GREY, alpha=0.7)
    ax.axhline(y=target, color=CE_ORANGE, linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title("1,000 Open Issues", color="white", fontsize=11)
    ax.set_ylabel("Open Issues", color=CE_LIGHT_GREY, fontsize=9)
    ax.annotate(
        f"At this rate... ~{target_date.strftime('%b %Y')}",
        xy=(target_date, target),
        color=CE_ORANGE,
        fontsize=9,
        ha="center",
        va="bottom",
    )

    # -- Panel 3: TypeScript 100% --
    ax = axes[1, 0]
    dates, vals = [], []
    for r in records:
        if "languages" not in r:
            continue
        langs = r["languages"]
        total = sum(langs.values())
        if total == 0:
            continue
        dt = datetime.datetime.fromisoformat(r["as_of"])
        # Only use post-mid-2023 data where TS migration is complete
        if dt < datetime.datetime(2023, 6, 1):
            continue
        dates.append(dt)
        vals.append(langs.get("TypeScript", 0) / total * 100)
    x = to_days(dates)
    coeffs = np.polyfit(x, vals, 1)
    slope_per_year = coeffs[0] * 365.25

    ax.plot(dates, vals, linewidth=1.5, color=CE_GREEN)

    # Cap the trend line to ~3 years out so the chart stays readable
    cap_end = to_days([dates[-1] + datetime.timedelta(days=3 * 365)])[0]
    trend_x = np.linspace(x[0], cap_end, 200)
    trend_y = np.polyval(coeffs, trend_x)
    trend_dates = [from_days(d) for d in trend_x]
    ax.plot(trend_dates, trend_y, "--", linewidth=1, color=CE_LIGHT_GREY, alpha=0.7)
    ax.axhline(y=100, color=CE_ORANGE, linestyle=":", linewidth=1, alpha=0.6)

    if coeffs[0] > 0.0001:
        target_day = (100 - coeffs[1]) / coeffs[0]
        target_date = from_days(target_day)
        msg = f"100% TypeScript: ~{target_date.strftime('%b %Y')} (at {slope_per_year:+.2f}%/yr)"
    elif coeffs[0] < -0.0001:
        msg = f"TypeScript: peaked. ({slope_per_year:+.1f}%/year)"
    else:
        msg = f"TypeScript: holding steady (~{vals[-1]:.0f}%)"

    ax.set_title("TypeScript 100%", color="white", fontsize=11)
    ax.set_ylabel("TypeScript %", color=CE_LIGHT_GREY, fontsize=9)
    ax.annotate(msg, xy=(0.5, 0.05), xycoords="axes fraction", color=CE_ORANGE, fontsize=8, ha="center")

    # -- Panel 4: Bug-free --
    ax = axes[1, 1]
    dates, vals = [], []
    for r in records:
        if "issues" not in r:
            continue
        bugs = r["issues"].get("open", {}).get("bug", 0)
        dates.append(datetime.datetime.fromisoformat(r["as_of"]))
        vals.append(bugs)
    x = to_days(dates)
    coeffs = np.polyfit(x, vals, 1)
    # Extrapolate backwards to find when bugs = 0
    zero_day = -coeffs[1] / coeffs[0]
    zero_date = from_days(zero_day)

    past_start = to_days([zero_date - datetime.timedelta(days=90)])[0]
    trend_x = np.linspace(past_start, x[-1], 200)
    trend_y = np.polyval(coeffs, trend_x)
    trend_dates = [from_days(d) for d in trend_x]

    ax.plot(dates, vals, linewidth=1.5, color=CE_RED)
    ax.plot(trend_dates, trend_y, "--", linewidth=1, color=CE_LIGHT_GREY, alpha=0.7)
    ax.axhline(y=0, color=CE_ORANGE, linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title("Bug-Free CE", color="white", fontsize=11)
    ax.set_ylabel("Open Bugs", color=CE_LIGHT_GREY, fontsize=9)
    ax.annotate(
        f"Last bug-free: ~{zero_date.strftime('%Y')}. It's been\ndownhill ever since.",
        xy=(0.5, 0.05),
        xycoords="axes fraction",
        color=CE_ORANGE,
        fontsize=8,
        ha="center",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out, "extrapolations.png")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def graph(input_file: str, output_dir: str) -> None:
    """Generate graphs from collected stats."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8")

    records = load_stats(input_file)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_open_issues(records, out)
    plot_stars_and_forks(records, out)
    plot_languages(records, out)
    plot_issue_categories(records, out)
    plot_extrapolations(records, out)


if __name__ == "__main__":
    cli()
