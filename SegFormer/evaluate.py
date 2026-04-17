"""
evaluate.py
===========
Run with NO arguments to produce all outputs automatically:

    python evaluate.py

What it saves (all under outputs/):
  • metrics_val.json          – precision / recall / IoU / mIoU / pixel-acc
  • loss_curve.png            – 3-panel training-history plot
  • classwise_overlay.png     – per-class semantic overlay on a sample val image

Optional overrides (all have safe defaults):
    --split       val | train | test          (default: val)
    --ckpt        path to checkpoint          (default: runs/segformer_change_best.pt)
    --batch_size  int                         (default: config.BATCH_SIZE)
    --img_size    int                         (default: config.IMG_SIZE)
    --history     path to train_history.json  (default: runs/train_history.json)
    --out_dir     directory for all outputs   (default: outputs/)
    --sample_idx  dataset index for overlay   (default: 0)
    --no_loss     skip loss-curve plot
    --no_overlay  skip classwise overlay
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from models.segformer_model import SegFormerChange
from utils.dataset import ChangeDataset
from utils.metrics import update_confusion_matrix, compute_metrics
from utils.visualtization import save_classwise_overlay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint(model, ckpt_path):
	try:
		state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
	except TypeError:
		state = torch.load(ckpt_path, map_location="cpu")
	if isinstance(state, dict) and "model_state" in state:
		model.load_state_dict(state["model_state"], strict=False)
	else:
		model.load_state_dict(state, strict=False)


# ---------------------------------------------------------------------------
# 1. Model Evaluation  →  metrics_<split>.json
# ---------------------------------------------------------------------------

def run_evaluation(split, ckpt_path, batch_size, img_size, out_dir):
	"""Evaluate the checkpoint on *split* and save a metrics JSON.

	Returns the metrics dict and the loaded model (reused for overlay).
	"""
	dataset = ChangeDataset(
		config.get_split_dir(split),
		img_size=img_size,
		num_classes=config.NUM_CLASSES,
		label_values=config.LABEL_VALUES,
	)
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=config.NUM_WORKERS,
		pin_memory=config.PIN_MEMORY and config.DEVICE == "cuda",
	)

	model = SegFormerChange(num_classes=config.NUM_CLASSES)
	load_checkpoint(model, ckpt_path)
	model.to(config.DEVICE)
	model.eval()

	conf_matrix = torch.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=torch.long)

	with torch.no_grad():
		for imgs, labels in loader:
			imgs   = imgs.to(config.DEVICE)
			labels = labels.to(config.DEVICE)
			outputs = model(imgs)
			outputs = F.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)
			preds   = torch.argmax(outputs, dim=1)
			conf_matrix = update_confusion_matrix(conf_matrix, labels.cpu(), preds.cpu(), config.NUM_CLASSES)

	metrics = compute_metrics(conf_matrix)

	out_path = Path(out_dir) / f"metrics_{split}.json"
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)
	print(f"[evaluate] Saved → {out_path}")
	print(json.dumps(metrics, indent=2))

	return metrics, model


# ---------------------------------------------------------------------------
# 2. Loss / Training-history plot  →  loss_curve.png
# ---------------------------------------------------------------------------

def plot_loss(history_path, out_path):
	"""Read train_history.json and save a 3-panel training-history figure."""
	import matplotlib.pyplot as plt
	import matplotlib.ticker as mticker

	history_path = Path(history_path)
	if not history_path.exists():
		print(f"[plot_loss] WARNING: history file not found at {history_path}. Skipping loss plot.")
		print("           Run train.py first – it writes runs/train_history.json automatically.")
		return

	with open(history_path, "r", encoding="utf-8") as f:
		history = json.load(f)

	if not history:
		print("[plot_loss] WARNING: history file is empty. Skipping.")
		return

	epochs      = [r["epoch"]                for r in history]
	train_loss  = [r["train_loss"]           for r in history]
	val_miou    = [r.get("val_miou", 0)      for r in history]
	val_pix_acc = [r.get("val_pixel_acc", 0) for r in history]

	best_idx  = int(max(range(len(val_miou)), key=lambda i: val_miou[i]))
	best_ep   = epochs[best_idx]
	best_loss = train_loss[best_idx]
	best_miou = val_miou[best_idx]
	best_acc  = val_pix_acc[best_idx]

	plt.style.use("dark_background")
	ACCENT   = "#4FC3F7"
	GREEN    = "#66BB6A"
	AMBER    = "#FFA726"
	BEST_CLR = "#EF5350"

	fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
	fig.patch.set_facecolor("#0D1117")

	for ax in axes:
		ax.set_facecolor("#161B22")
		ax.tick_params(colors="#C9D1D9", labelsize=9)
		ax.xaxis.label.set_color("#C9D1D9")
		ax.yaxis.label.set_color("#C9D1D9")
		ax.title.set_color("#E6EDF3")
		for spine in ax.spines.values():
			spine.set_edgecolor("#30363D")
		ax.grid(True, color="#21262D", linewidth=0.8, linestyle="--")
		ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

	def _best_vline(ax):
		ax.axvline(best_ep, color=BEST_CLR, linewidth=1.4, linestyle=":",
				   alpha=0.85, label=f"Best epoch ({best_ep})")

	# Panel 1 – training loss
	ax = axes[0]
	ax.plot(epochs, train_loss, color=ACCENT, linewidth=2, marker="o", markersize=3.5, label="Train loss")
	ax.plot(best_ep, best_loss, "o", color=BEST_CLR, markersize=8, zorder=5)
	_best_vline(ax)
	ax.set_title("Training Loss", fontsize=13, fontweight="bold", pad=10)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Cross-Entropy Loss")
	ax.legend(fontsize=8, framealpha=0.4)

	# Panel 2 – val mIoU
	ax = axes[1]
	ax.plot(epochs, val_miou, color=GREEN, linewidth=2, marker="s", markersize=3.5, label="Val mIoU")
	ax.plot(best_ep, best_miou, "s", color=BEST_CLR, markersize=8, zorder=5)
	_best_vline(ax)
	ax.set_title("Validation mIoU", fontsize=13, fontweight="bold", pad=10)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("mIoU")
	ax.set_ylim(bottom=0)
	ax.legend(fontsize=8, framealpha=0.4)

	# Panel 3 – val pixel accuracy
	ax = axes[2]
	ax.plot(epochs, val_pix_acc, color=AMBER, linewidth=2, marker="^", markersize=3.5, label="Val Pixel Acc")
	ax.plot(best_ep, best_acc, "^", color=BEST_CLR, markersize=8, zorder=5)
	_best_vline(ax)
	ax.set_title("Validation Pixel Accuracy", fontsize=13, fontweight="bold", pad=10)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Pixel Accuracy")
	ax.set_ylim(bottom=0)
	ax.legend(fontsize=8, framealpha=0.4)

	summary = (
		f"Best epoch {best_ep}  │  "
		f"loss {best_loss:.4f}  │  "
		f"mIoU {best_miou:.4f}  │  "
		f"pix-acc {best_acc:.4f}"
	)
	fig.suptitle(
		f"Training History  –  {len(epochs)} epochs\n{summary}",
		fontsize=12, fontweight="bold", color="#E6EDF3", y=1.03,
	)

	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
	plt.close(fig)
	print(f"[plot_loss] Saved → {out_path}")


# ---------------------------------------------------------------------------
# 3. Per-class semantic overlay  →  classwise_overlay.png
# ---------------------------------------------------------------------------

def run_classwise_overlay(model, split, img_size, sample_idx, out_path):
	"""Run inference on one sample from *split* and save the classwise overlay."""
	dataset = ChangeDataset(
		config.get_split_dir(split),
		img_size=img_size,
		num_classes=config.NUM_CLASSES,
		label_values=config.LABEL_VALUES,
		return_name=True,
	)
	if len(dataset) == 0:
		print(f"[classwise] WARNING: {split} split is empty. Skipping overlay.")
		return

	idx = min(sample_idx, len(dataset) - 1)
	_, _, name = dataset[idx]

	im1_path = str(config.get_split_dir(split) / "im1" / name)
	im2_path = str(config.get_split_dir(split) / "im2" / name)

	im1 = cv2.imread(im1_path)
	im2 = cv2.imread(im2_path)
	if im1 is None or im2 is None:
		print(f"[classwise] WARNING: could not read images for sample '{name}'. Skipping overlay.")
		return

	im1 = cv2.resize(im1, (img_size, img_size))
	im2 = cv2.resize(im2, (img_size, img_size))

	img_tensor = np.concatenate([im1 / 255.0, im2 / 255.0], axis=2)
	img_tensor = torch.tensor(img_tensor).permute(2, 0, 1).float().unsqueeze(0).to(config.DEVICE)

	model.eval()
	with torch.no_grad():
		logits = model(img_tensor)
		logits = F.interpolate(logits, size=(img_size, img_size), mode="bilinear", align_corners=False)
		pred   = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype("uint8")

	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	save_classwise_overlay(
		im1, im2, pred, out_path,
		num_classes=config.NUM_CLASSES,
		class_names=config.CLASS_NAMES,
	)


# ---------------------------------------------------------------------------
# main – runs everything by default
# ---------------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser(
		description="Evaluate model and save all outputs. Run with no args for full auto mode."
	)
	parser.add_argument("--split",      default="val",  choices=["train", "val", "test"])
	parser.add_argument("--ckpt",       default=str(config.CHECKPOINT_PATH))
	parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
	parser.add_argument("--img_size",   type=int, default=config.IMG_SIZE)
	parser.add_argument("--history",    default=str(config.SAVE_DIR / "train_history.json"))
	parser.add_argument("--out_dir",    default=str(config.BASE_DIR / "outputs"))
	parser.add_argument("--sample_idx", type=int, default=0,
		help="Dataset index of the sample used for the classwise overlay (default: 0)")
	parser.add_argument("--no_loss",    action="store_true", help="Skip the loss-curve plot")
	parser.add_argument("--no_overlay", action="store_true", help="Skip the classwise semantic overlay")
	args = parser.parse_args()

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	print("\n" + "="*60)
	print("  SegFormer Evaluate  –  auto saving all outputs to:", out_dir)
	print("="*60 + "\n")

	# ── 1. Evaluate & save metrics JSON ────────────────────────────────────
	print("── Step 1/3: Model evaluation ──")
	_, model = run_evaluation(
		args.split, args.ckpt, args.batch_size, args.img_size, out_dir
	)

	# ── 2. Loss curve ───────────────────────────────────────────────────────
	if not args.no_loss:
		print("\n── Step 2/3: Loss curve ──")
		plot_loss(args.history, out_dir / "loss_curve.png")
	else:
		print("\n── Step 2/3: Loss curve  [SKIPPED via --no_loss] ──")

	# ── 3. Per-class overlay ────────────────────────────────────────────────
	if not args.no_overlay:
		print("\n── Step 3/3: Classwise semantic overlay ──")
		run_classwise_overlay(
			model, args.split, args.img_size,
			args.sample_idx,
			out_dir / "classwise_overlay.png",
		)
	else:
		print("\n── Step 3/3: Classwise overlay  [SKIPPED via --no_overlay] ──")

	print("\n" + "="*60)
	print("  All outputs saved to:", out_dir)
	print("="*60 + "\n")


if __name__ == "__main__":
	main()
