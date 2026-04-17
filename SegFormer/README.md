# SegFormer Change Detection Project

This repository contains a PyTorch implementation of a SegFormer-based model for image change detection, designed for Ubuntu. The model uses the `nvidia/segformer-b5-finetuned-ade-640-640` architecture and detects multiple classes of changes between two co-registered images (`im1` and `im2`).

## Prerequisites

- **Ubuntu 20.04 or 22.04** (Recommended)
- **Python 3.8+**
- (Optional but recommended) **NVIDIA GPU** with CUDA installed for accelerated training and inference.

## Project Setup on Ubuntu

### 1. System Updates and Dependencies

First, ensure your system is up to date and you have the necessary base dependencies:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git
```

*(Optional)* Some OpenCV features may require extra system libraries. If you run into issues running `cv2`, run:
```bash
sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

### 2. Prepare the Python Environment

Navigate to the directory where you want to set up the project (or inside the copied project folder) and create a virtual environment:

```bash
# Navigate to the project directory
cd SegFormer

# Create a virtual environment named `.venv`
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 3. Install Python Dependencies

With the virtual environment active, install the necessary Python packages:

```bash
# Upgrade pip
pip install --upgrade pip

# Install project requirements
pip install -r requirements.txt
```

*(Note: The PyTorch version installed via `requirements.txt` might be the CPU-only version depending on your pip configuration. To install a GPU-accelerated PyTorch for Ubuntu, visit the [PyTorch Get Started](https://pytorch.org/get-started/locally/) page. Example for CUDA 11.8:)*
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Hugging Face Authentication

The project downloads pre-trained weights (`nvidia/segformer-b5-finetuned-ade-640-640`) from Hugging Face. To avoid rate limits or access issues, log in with your Hugging Face access token:

1. Create a token from your [Hugging Face settings](https://huggingface.co/settings/tokens).
2. Run the CLI login command in your terminal:
   ```bash
   huggingface-cli login
   ```
3. Paste your token when prompted.

### 5. Data Preparation

Place your data into the appropriate directories before training. The project reads data from `data/raw/` or `data/processed/`.

Directory structure should look like this:
```text
data/
  raw/
    im1/      # Pre-change images
    im2/      # Post-change images
    label1/   # Labels (if applicable)
    label2/   # Labels (if applicable)
```
*(Check `utils/preprocessing.py` or the dataset script if you need to run any local preprocessing to generate `data/processed/` outputs).*

## Usage

Ensure you are always in your virtual environment (`source .venv/bin/activate`) before running the scripts.

- **Start Training:**
  ```bash
  python train.py
  ```

- **Evaluate Model:**
  ```bash
  python evaluate.py
  ```

- **Run Inference/Testing:**
  ```bash
  python test.py
  ```

Model checkpoints and run history will be saved in the `runs/` directory (e.g., `segformer_change_best.pt` and `train_history.json`). Outputs and visualizations from testing will go to the `outputs/` folder.
