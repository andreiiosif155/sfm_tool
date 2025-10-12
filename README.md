# SFM Tool

## Installation

```
conda create -n sfmtool -y python=3.9
conda activate sfmtool

git clone --recursive https://github.com/andreiiosif155/sfm_tool.git
cd sfm_tool

pip install -r requirements.txt
pip install -e third_party/Hierarchical-Localization
```
## Polycam Exporter

```
There are **two ways to run the exporter**:

1. Auto-run after SfM
   Set:
   EXPORT_POLYCAM_BINS=1
   POLYCAM_NUM_INLIERS= none or adaptive or all.
   Run `sfm_tool.py`. The exporter will read
   `out_dir/colmap/features.h5` and `out_dir/colmap/matches.h5`, then write
   `out_dir/polycam_bins/`.
  Command example:
    EXPORT_POLYCAM_BINS=1 \
    POLYCAM_NUM_INLIERS=all \
    python3 sfm_tool.py \
      --data images \
      --output-dir out_sp256 \
      --sfm-tool hloc \
      --feature-type superpoint \
      --matcher-type superpoint+lightglue \
      --skip-image-processing

2. Standalone exporter script
   Run this after you already have `features.h5` and `matches.h5`
   Command example:
     python3 utils/polycam_exporter.py \
      --sparse out_test/colmap/sparse/0 \
      --features-h5 out_test/colmap/features.h5 \
      --matches-h5 out_test/colmap/matches.h5 \
      --images images \
      --dst out_test/polycam_bins \
      --descriptor-dim 256 \
      --num-inliers all
