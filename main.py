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


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def graph(input_file: str, output_dir: str) -> None:
    """Generate graphs from collected stats."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    CE_GREEN = "#67c52a"
    CE_DARK = "#282828"
    CE_LIGHT_GREY = "#aaaaaa"

    dates = []
    counts = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "as_of" not in record or "open_issues_count" not in record:
                continue
            dates.append(datetime.datetime.fromisoformat(record["as_of"]))
            counts.append(record["open_issues_count"])

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.set_facecolor(CE_DARK)
    ax.set_facecolor(CE_DARK)
    ax.plot(dates, counts, linewidth=1.5, color=CE_GREEN)
    ax.set_title("Compiler Explorer — Open Issues Over Time", color="white", fontsize=14)
    ax.set_xlabel("Date", color=CE_LIGHT_GREY)
    ax.set_ylabel("Open Issues", color=CE_LIGHT_GREY)
    ax.tick_params(colors=CE_LIGHT_GREY)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.grid(True, color=CE_LIGHT_GREY, alpha=0.2)
    for spine in ax.spines.values():
        spine.set_color(CE_LIGHT_GREY)
        spine.set_alpha(0.3)
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "open-issues.png", dpi=150, facecolor=CE_DARK)
    plt.close(fig)
    click.echo(f"Saved {out / 'open-issues.png'}")


if __name__ == "__main__":
    cli()
