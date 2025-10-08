from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Literal

from rich.table import Table

from sfm_tool import SfM
from utils.rich_utils import CONSOLE


DEFAULT_COMBINATIONS: Tuple[str, ...] = (
    "superpoint_aachen:superglue",
    "superpoint_aachen:superglue-fast",
    "superpoint_aachen:superpoint+lightglue",
    "superpoint_aachen:NN-superpoint",
    "superpoint_max:superglue",
    "superpoint_max:superpoint+lightglue",
    "superpoint_max:NN-superpoint",
    "superpoint_inloc:superglue",
    "superpoint_inloc:superpoint+lightglue",
    "superpoint_inloc:NN-superpoint",
    "r2d2:NN",
    "d2net-ss:NN",
    "sosnet:NN",
    "disk:disk+lightglue",
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
    if colmap_dir.exists():
        if not reuse_intermediate:
            shutil.rmtree(colmap_dir)
        else:
            sparse_dir = colmap_dir / "sparse" / "0"
            if sparse_dir.exists():
                CONSOLE.print(
                    f"[bold yellow]Skipping[/] {feature_type}/{matcher_type} "
                    f"(existing reconstruction found at {colmap_dir / 'sparse' / '0'})"
                )
                # Load the existing COLMAP reconstruction using pycolmap.
                import pycolmap
                reconstruction = pycolmap.Reconstruction(str(sparse_dir))
                CONSOLE.print(
                    f"[bold green]Loaded reconstruction from {sparse_dir} using pycolmap.[/]"
                )

                # Extract evaluation metrics from the loaded reconstruction.
                stats = {
                    "num_points3D": reconstruction.num_points3D(),
                    "mean_track_length": reconstruction.compute_mean_track_length(),
                    "mean_observations_per_image": reconstruction.compute_mean_observations_per_reg_image(),
                    "mean_reprojection_error": reconstruction.compute_mean_reprojection_error(),
                }

                return EvaluationResult(
                    feature_type=feature_type,
                    matcher_type=matcher_type,
                    output_dir=combo_output_dir,
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

    if not stats:
        CONSOLE.print(
            f"[bold yellow]No evaluation statistics returned for[/] "
            f"{feature_type}/{matcher_type}. Skipping."
        )
        return None

    return EvaluationResult(
        feature_type=feature_type,
        matcher_type=matcher_type,
        output_dir=combo_output_dir,
        stats=stats,
    )


def _display_results(results: List[EvaluationResult]) -> None:
    """Pretty-print the sorted evaluation results."""
    table = Table(title="SfM evaluation results (best first)", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("Combination")
    table.add_column("Mean reproj. err", justify="right")
    table.add_column("Points3D", justify="right")
    table.add_column("Mean track len", justify="right")
    table.add_column("Obs / image", justify="right")
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
            str(result.output_dir / "colmap" / "sparse" / "0"),
        )

    CONSOLE.print(table)


def _export_results(results: Iterable[EvaluationResult], destination: Path) -> None:
    serializable = [
        {
            "feature_type": result.feature_type,
            "matcher_type": result.matcher_type,
            "output_dir": str(result.output_dir),
            **result.stats,
        }
        for result in results
    ]
    destination.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    CONSOLE.print(f"[bold green]Saved evaluation summary to[/] {destination}")


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
