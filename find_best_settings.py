from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Literal

from rich.table import Table

from sfm_tool import SfM
from utils.rich_utils import CONSOLE


DEFAULT_COMBINATIONS: Tuple[str, ...] = (
    "sift:NN-superpoint",
    "sift:NN-ratio",
    "sift:NN-mutual",
    "sift:adalam",
    "superpoint_aachen:superglue",
    "superpoint_aachen:superglue-fast",
    "superpoint_aachen:NN-superpoint",
    "superpoint_aachen:NN-ratio",
    "superpoint_aachen:NN-mutual",
    "superpoint_aachen:superpoint+lightglue",
    "superpoint_max:superglue",
    "superpoint_max:superglue-fast",
    "superpoint_max:NN-superpoint",
    "superpoint_max:NN-ratio",
    "superpoint_max:NN-mutual",
    "superpoint_max:superpoint+lightglue",
    "superpoint_inloc:superglue",
    "superpoint_inloc:superglue-fast",
    "superpoint_inloc:NN-superpoint",
    "superpoint_inloc:NN-ratio",
    "superpoint_inloc:NN-mutual",
    "superpoint_inloc:superpoint+lightglue",
    "r2d2:NN-superpoint",
    "r2d2:NN-ratio",
    "r2d2:NN-mutual",
    "sosnet:NN",
    "sosnet:NN-superpoint",
    "sosnet:NN-ratio",
    "sosnet:NN-mutual",
    "sosnet:adalam",
    "sosnet:disk+lightglue",
    "disk:NN-superpoint",
    "disk:NN-ratio",
    "disk:NN-mutual",
    "disk:disk+lightglue",
    "aliked-n16:superglue",
    "aliked-n16:NN-mutual",
    "aliked-n16:aliked+lightglue",
)

SLUG_SANITIZER = re.compile(r"[^A-Za-z0-9_]+")


def _parse_combinations(raw_items: Sequence[str]) -> List[Tuple[str, str]]:
    """Parse a list of feature/matcher specification strings."""
    combinations: List[Tuple[str, str]] = []
    for raw in raw_items:
        item = raw.strip()
        if not item:
            continue
        separator = ":" if ":" in item else ","
        if separator not in item:
            raise ValueError(
                f"Invalid combination '{raw}'. Expected format 'feature:matcher'."
            )
        feature, matcher = (part.strip() for part in item.split(separator, 1))
        if not feature or not matcher:
            raise ValueError(
                f"Invalid combination '{raw}'. Expected format 'feature:matcher'."
            )
        combinations.append((feature, matcher))
    if not combinations:
        raise ValueError("No valid feature/matcher combinations provided.")
    return combinations


def _slugify(feature: str, matcher: str) -> str:
    """Create a filesystem-friendly name for a feature/matcher combination."""
    raw_slug = f"{feature}_{matcher.replace('+', '_')}"
    sanitized = SLUG_SANITIZER.sub("_", raw_slug)
    sanitized = sanitized.strip("_")
    return re.sub(r"_+", "_", sanitized)


@dataclass
class EvaluationResult:
    feature_type: str
    matcher_type: str
    output_dir: Path
    model_dir: Path
    stats: Dict[str, float]

    @property
    def combination_name(self) -> str:
        return _slugify(self.feature_type, self.matcher_type)


@dataclass
class Args:
    """Configuration options for sweeping SfM feature/matcher combinations."""

    images_dir: Path
    """Path to the directory containing the input images."""

    output_root: Path = Path("outputs/find_best_settings")
    """Directory where per-combination results will be stored."""

    combinations: Optional[List[str]] = None
    """Feature/matcher combinations in the format feature:matcher."""

    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree"
    """Image matching strategy used by hloc."""

    reuse_intermediate: bool = True
    """Keep existing COLMAP outputs instead of recreating them."""

    verbose: bool = False
    """Enable verbose logging for the SfM pipeline."""

    export_figure: Optional[Path] = None
    """Path to save the bar chart summary. Defaults to `<output_root>/metrics_summary.png`."""

    export_json: Optional[Path] = None
    """Optional path to save the sorted evaluation metrics as JSON."""


def _sort_key(result: EvaluationResult) -> Tuple[float, float, float, float]:
    """Sorting key implementing the prioritization rules."""
    stats = result.stats
    reproj = float(stats.get("mean_reprojection_error", float("inf")) or float("inf"))
    num_pts = float(stats.get("num_points3D", 0.0) or 0.0)
    track_len = float(stats.get("mean_track_length", 0.0) or 0.0)
    obs_per_img = float(stats.get("mean_observations_per_image", 0.0) or 0.0)
    return (reproj, -num_pts, -track_len, -obs_per_img)


def _normalize_stats(stats: Dict[str, Any]) -> Dict[str, float]:
    """Convert raw stats into floats, skipping None entries."""
    normalized: Dict[str, float] = {}
    for key, value in stats.items():
        if value is None:
            continue
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _find_colmap_model_dirs(colmap_dir: Path) -> List[Path]:
    """Return directories that appear to contain a COLMAP reconstruction."""
    if not colmap_dir.exists():
        return []
    candidates = set()
    for ext in ("bin", "txt"):
        for cameras_file in colmap_dir.rglob(f"cameras.{ext}"):
            parent = cameras_file.parent
            if (
                (parent / f"images.{ext}").exists()
                and (parent / f"points3D.{ext}").exists()
            ):
                candidates.add(parent)
    return sorted(candidates)


def _load_reconstruction_stats(
    model_dir: Path, verbose: bool = False
) -> Optional[Dict[str, float]]:
    """Load a reconstruction and extract evaluation metrics."""
    try:
        import pycolmap
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pycolmap is required to inspect existing COLMAP reconstructions."
        ) from exc

    try:
        reconstruction = pycolmap.Reconstruction(str(model_dir))
    except Exception as exc:  # pragma: no cover
        if verbose:
            CONSOLE.print(
                f"[bold red]Failed to load reconstruction at {model_dir}: {exc}[/]"
            )
        return None

    return {
        "num_points3D": float(reconstruction.num_points3D()),
        "mean_track_length": float(reconstruction.compute_mean_track_length()),
        "mean_observations_per_image": float(
            reconstruction.compute_mean_observations_per_reg_image()
        ),
        "mean_reprojection_error": float(
            reconstruction.compute_mean_reprojection_error()
        ),
        "num_reg_images": float(reconstruction.num_reg_images()),
    }


def _select_best_reconstruction(
    colmap_dir: Path, verbose: bool = False
) -> Optional[Tuple[Path, Dict[str, float]]]:
    """Return the reconstruction with the highest number of registered images."""
    best: Optional[Tuple[Path, Dict[str, float]]] = None
    for model_dir in _find_colmap_model_dirs(colmap_dir):
        stats = _load_reconstruction_stats(model_dir, verbose=verbose)
        if stats is None:
            continue
        if best is None or stats.get("num_reg_images", 0.0) > best[1].get(
            "num_reg_images", 0.0
        ):
            best = (model_dir, stats)
    return best


def _run_combination(
    images_dir: Path,
    output_root: Path,
    feature_type: str,
    matcher_type: str,
    matching_method: str,
    verbose: bool,
    reuse_intermediate: bool,
) -> Optional[EvaluationResult]:
    combination_name = _slugify(feature_type, matcher_type)
    combo_output_dir = output_root / combination_name
    combo_output_dir.mkdir(parents=True, exist_ok=True)

    # Reset the COLMAP directory when requested to ensure fresh reconstructions.
    colmap_dir = combo_output_dir / "colmap"
    if colmap_dir.exists() and not reuse_intermediate:
        shutil.rmtree(colmap_dir)

    best_existing: Optional[Tuple[Path, Dict[str, float]]] = None
    if reuse_intermediate:
        best_existing = _select_best_reconstruction(colmap_dir, verbose=verbose)
        if best_existing is not None:
            model_dir, stats = best_existing
            stats = _normalize_stats(stats)
            CONSOLE.print(
                f"[bold yellow]Reusing[/] {feature_type}/{matcher_type} "
                f"(selected existing model at {model_dir} with {int(stats.get('num_reg_images', 0))} registered images)"
            )
            return EvaluationResult(
                feature_type=feature_type,
                matcher_type=matcher_type,
                output_dir=combo_output_dir,
                model_dir=model_dir,
                stats=stats,
            )

    CONSOLE.print(
        f"[bold cyan]Evaluating[/] {feature_type} / {matcher_type} "
        f"→ results in {combo_output_dir}"
    )
    try:
        sfm = SfM(
            data=images_dir,
            output_dir=combo_output_dir,
            sfm_tool="hloc",
            feature_type=feature_type,
            matcher_type=matcher_type,
            matching_method=matching_method,
            skip_image_processing=True,
            verbose=verbose,
        )
        stats = sfm.run()
    except Exception as exc:  # Catch broad exceptions to keep the sweep running.
        CONSOLE.print(
            f"[bold red]Failed[/] {feature_type}/{matcher_type}: {exc}",
            highlight=False,
        )
        if verbose:
            raise
        return None

    best_after_run = _select_best_reconstruction(colmap_dir, verbose=verbose)
    if best_after_run is not None:
        model_dir, selected_stats = best_after_run
        selected_stats = _normalize_stats(selected_stats)
        CONSOLE.print(
            f"[bold green]Completed[/] {feature_type}/{matcher_type} "
            f"(using model at {model_dir} with {int(selected_stats.get('num_reg_images', 0))} registered images)"
        )
        return EvaluationResult(
            feature_type=feature_type,
            matcher_type=matcher_type,
            output_dir=combo_output_dir,
            model_dir=model_dir,
            stats=selected_stats,
        )

    if stats:
        stats = _normalize_stats(stats)
        fallback_dir = colmap_dir / "sparse" / "0"
        if "num_reg_images" not in stats:
            # Attempt to enrich the stats with reconstruction metadata.
            loaded = _load_reconstruction_stats(fallback_dir, verbose=verbose)
            if loaded is not None:
                stats.update(_normalize_stats(loaded))
        return EvaluationResult(
            feature_type=feature_type,
            matcher_type=matcher_type,
            output_dir=combo_output_dir,
            model_dir=fallback_dir,
            stats=stats,
        )

    CONSOLE.print(
        f"[bold yellow]No evaluation statistics returned for[/] "
        f"{feature_type}/{matcher_type}. Skipping."
    )
    return None


def _display_results(results: List[EvaluationResult]) -> None:
    """Pretty-print the sorted evaluation results."""
    table = Table(title="SfM evaluation results (best first)", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("Combination")
    table.add_column("Mean reproj. err", justify="right")
    table.add_column("Points3D", justify="right")
    table.add_column("Mean track len", justify="right")
    table.add_column("Obs / image", justify="right")
    table.add_column("Reg. images", justify="right")
    table.add_column("Output path")

    for idx, result in enumerate(results, start=1):
        stats = result.stats
        table.add_row(
            str(idx),
            f"{result.feature_type}/{result.matcher_type}",
            f"{stats.get('mean_reprojection_error', float('nan')):.4f}",
            f"{int(stats.get('num_points3D', 0)):,}",
            f"{stats.get('mean_track_length', 0.0):.2f}",
            f"{stats.get('mean_observations_per_image', 0.0):.2f}",
            f"{int(stats.get('num_reg_images', 0)):,}",
            str(result.model_dir),
        )

    CONSOLE.print(table)


def _export_results(results: Iterable[EvaluationResult], destination: Path) -> None:
    serializable = [
        {
            "feature_type": result.feature_type,
            "matcher_type": result.matcher_type,
            "output_dir": str(result.output_dir),
            "model_dir": str(result.model_dir),
            **result.stats,
        }
        for result in results
    ]
    destination.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    CONSOLE.print(f"[bold green]Saved evaluation summary to[/] {destination}")


def _save_metrics_figure(results: List[EvaluationResult], destination: Path) -> None:
    if not results:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        CONSOLE.print(
            "[bold yellow]matplotlib not available; skipping metric figure export.[/]"
        )
        return

    labels = [f"{res.feature_type} / {res.matcher_type}" for res in results]
    metrics = [
        ("num_reg_images", "Registered images"),
        ("num_points3D", "3D points"),
        ("mean_reprojection_error", "Mean reprojection error"),
        ("mean_track_length", "Mean track length"),
    ]

    # Horizontal bar charts scale better with many combinations.
    height = len(results) * 0.35
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(14, height),
        constrained_layout=True,
        sharey=True,
    )

    # Ensure axes is iterable when len(metrics)==1 (not the case now but defensive).
    if hasattr(axes, "flatten"):
        axes = list(axes.flatten())
    elif isinstance(axes, (list, tuple)):
        axes = list(axes)
    else:
        axes = [axes]

    for ax, (key, title) in zip(axes, metrics):
        values = [float(res.stats.get(key, float("nan"))) for res in results]
        ax.barh(range(len(results)), values, color="#4C72B0")
        ax.set_title(title)
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.invert_yaxis()  # Top bar corresponds to the best result in the list order.
        if key in ("num_reg_images", "num_points3D"):
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)

    fig.suptitle("SfM metrics comparison", fontsize=14)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    CONSOLE.print(f"[bold green]Saved metrics figure to[/] {destination}")


def main(args: Args) -> int:
    """Entry point for the find-best-settings sweep."""
    images_dir = args.images_dir.expanduser().resolve()
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(
            f"Image directory '{images_dir}' does not exist or is not a directory."
        )

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    raw_combinations = args.combinations or DEFAULT_COMBINATIONS
    try:
        combinations = _parse_combinations(raw_combinations)
    except ValueError as exc:
        raise SystemExit(str(exc))

    results: List[EvaluationResult] = []
    for feature_type, matcher_type in combinations:
        try:
            result = _run_combination(
                images_dir=images_dir,
                output_root=output_root,
                feature_type=feature_type,
                matcher_type=matcher_type,
                matching_method=args.matching_method,
                verbose=args.verbose,
                reuse_intermediate=args.reuse_intermediate,
            )
        except Exception as exc:
            CONSOLE.print(
                f"[bold red]Error while processing {feature_type}/{matcher_type}: {exc}",
                highlight=False,
            )
            continue

        if result is not None:
            results.append(result)

    if not results:
        CONSOLE.print(
            "[bold red]No successful reconstructions were produced. Nothing to rank."
        )
        return 1

    results.sort(key=_sort_key)
    _display_results(results)

    figure_path = args.export_figure
    if figure_path is None:
        figure_path = output_root / "metrics_summary.png"
    figure_path = figure_path.expanduser().resolve()
    _save_metrics_figure(results, figure_path)

    if args.export_json:
        export_path = args.export_json.expanduser().resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        _export_results(results, export_path)

    return 0


if __name__ == "__main__":
    import tyro
    tyro.extras.set_accent_color("bright_yellow")
    args = tyro.cli(Args)
    main(args)
