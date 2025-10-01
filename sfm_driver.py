#!/usr/bin/env python
"""Minimal CLI: process a video or image sequence into COLMAP reconstruction (database + sparse model)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import subprocess
import shutil
import tyro
import torch
from rich.console import Console

import process_data_utils
import colmap_utils
import hloc_utils

CONSOLE = Console()


@dataclass
class SfMConverter:
    """Process images or video into COLMAP SfM outputs."""

    data: Path
    """Path to folder of images or a video file (.mp4, .mov, .avi, .mkv)."""

    output_dir: Path = Path("colmap_output")
    """Where to save COLMAP results."""

    sfm_tool: Literal["colmap", "hloc"] = "hloc"
    """Which SfM tool to use (colmap=SIFT, hloc=SuperPoint+SuperGlue)."""

    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "exhaustive"
    """Feature matching strategy."""

    feature_type: str = "superpoint_aachen"
    matcher_type: str = "superglue"
    fps: int = 2
    """FPS for frame extraction if input is a video."""

    gpu: bool = torch.cuda.is_available()
    """Automatically detect GPU availability (fallback to CPU if not)."""

    verbose: bool = True

    def _extract_frames(self) -> Path:
        """Extract frames from a video into output_dir/images/"""
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found. Please install it (apt-get install ffmpeg).")

        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-i", str(self.data),
            "-vf", f"fps={self.fps}",
            str(image_dir / "%04d.jpg"),
        ]
        CONSOLE.log(f"🎥 Extracting frames with ffmpeg: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return image_dir

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # If video, extract frames first
        if self.data.is_file() and self.data.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
            image_dir = self._extract_frames()
        else:
            image_dir = self.data

        # Paths
        database_path = self.output_dir / "database.db"
        sparse_dir = self.output_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        # Run SfM
        if self.sfm_tool == "colmap":
            CONSOLE.log("[bold cyan]Running COLMAP pipeline...")
            colmap_utils.run_colmap(
                image_dir=image_dir,
                colmap_dir=self.output_dir,
                camera_model=process_data_utils.CAMERA_MODELS["perspective"],
                gpu=self.gpu,
                verbose=self.verbose,
                matching_method=self.matching_method,
                refine_intrinsics=True,
            )
        elif self.sfm_tool == "hloc":
            CONSOLE.log("[bold cyan]Running HLOC pipeline (SuperPoint + SuperGlue)...")
            hloc_utils.run_hloc(
                image_dir=image_dir,
                colmap_dir=self.output_dir,
                camera_model=process_data_utils.CAMERA_MODELS["perspective"],
                verbose=self.verbose,
                matching_method=self.matching_method,
                feature_type=self.feature_type,
                matcher_type=self.matcher_type,
            )
        else:
            raise RuntimeError(f"Unknown sfm_tool: {self.sfm_tool}")

        CONSOLE.log(f"✅ SfM finished. Results saved in [green]{self.output_dir}[/green]")
        CONSOLE.log(f"   - Database: {database_path}")
        CONSOLE.log(f"   - Sparse reconstruction: {sparse_dir}/0")


def entrypoint():
    tyro.cli(SfMConverter).run()


if __name__ == "__main__":
    entrypoint()

