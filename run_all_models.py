from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def run_improved():
    from early_warning.config import PipelineConfig
    from early_warning.improved_pipeline import run_improved_pipeline
    
    project_root = Path.cwd()
    if project_root.name == "Remote Sensing":
        project_root = project_root.parent
        
    print("=" * 60)
    print("IMPROVED GIS MODELLING PIPELINE (Classification)")
    print("=" * 60)
    config = PipelineConfig(project_root=project_root)
    run_improved_pipeline(config)


def run_standard():
    from early_warning.config import PipelineConfig
    from early_warning.pipeline import run_pipeline

    project_root = Path.cwd()
    if project_root.name == "Remote Sensing":
        project_root = project_root.parent

    print("=" * 60)
    print("STANDARD GIS MODELLING PIPELINE (Classification)")
    print("=" * 60)
    config = PipelineConfig(project_root=project_root)
    run_pipeline(config)


def run_regression():
    from early_warning.regression_pipeline import RegressionConfig, run_regression_pipeline

    project_root = Path.cwd()
    if project_root.name == "Remote Sensing":
        project_root = project_root.parent

    print("=" * 60)
    print("REGRESSION PIPELINE")
    print("=" * 60)
    config = RegressionConfig(project_root=project_root, use_gpu=True)
    run_regression_pipeline(config)


def main():
    parser = argparse.ArgumentParser(description="Run all GIS models from a single file.")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["improved", "standard", "regression", "all"], 
        default="all",
        help="Model pipeline to run (improved, standard, regression, all)"
    )
    args = parser.parse_args()

    if args.model in ("improved", "all"):
        run_improved()
    
    if args.model in ("standard", "all"):
        run_standard()
        
    if args.model in ("regression", "all"):
        run_regression()

if __name__ == "__main__":
    main()
