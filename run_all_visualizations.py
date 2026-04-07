from __future__ import annotations
import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def main():
    try:
        from early_warning.visualization import run_all_visualizations
        
        project_root = Path.cwd()
        if project_root.name == "Remote Sensing":
            project_root = project_root.parent
            
        print("=" * 60)
        print("RUNNING ALL VISUALIZATIONS")
        print("=" * 60)
        
        run_all_visualizations(project_root)
        
    except ImportError as e:
        print(f"Visualization methods are consolidated. Error: {e}")

if __name__ == "__main__":
    main()
