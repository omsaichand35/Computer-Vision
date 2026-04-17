import math

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import config


def get_palette(num_classes=7):
	palette = list(config.CLASS_COLORS_BGR)
	if num_classes <= len(palette):
		return palette[:num_classes]

	extra = np.random.randint(0, 255, size=(num_classes - len(palette), 3))
	palette.extend([tuple(map(int, c)) for c in extra])
	return palette


def colorize_mask(mask, num_classes=7):
	palette = get_palette(num_classes)
	h, w = mask.shape
	color = np.zeros((h, w, 3), dtype=np.uint8)
	for cls_id in range(num_classes):
		color[mask == cls_id] = palette[cls_id]
	return color


def overlay_mask(image_bgr, mask, alpha=0.5, num_classes=7):
	color_mask = colorize_mask(mask, num_classes)
	return cv2.addWeighted(image_bgr, 1 - alpha, color_mask, alpha, 0)


def save_prediction_triplet(im1, im2, pred_mask, out_path, num_classes=7):
	pred_color = colorize_mask(pred_mask, num_classes)
	combined = np.concatenate([im1, im2, pred_color], axis=1)
	cv2.imwrite(out_path, combined)


def create_legend_panel(class_names=None, num_classes=7, width=320, row_h=38):
	if class_names is None:
		class_names = list(config.CLASS_NAMES)

	class_names = class_names[:num_classes]
	height = max(80, row_h * len(class_names) + 30)
	panel = np.full((height, width, 3), 30, dtype=np.uint8)

	palette = get_palette(num_classes)
	cv2.putText(panel, "Legend", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

	for i, class_name in enumerate(class_names):
		y = 40 + i * row_h
		color = palette[i]
		cv2.rectangle(panel, (16, y), (46, y + 22), color, -1)
		cv2.rectangle(panel, (16, y), (46, y + 22), (220, 220, 220), 1)
		label = f"{i}: {class_name}"
		cv2.putText(panel, label, (58, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)

	return panel


def save_prediction_with_legend(im1, im2, pred_mask, out_path, num_classes=7, class_names=None):
	pred_color = colorize_mask(pred_mask, num_classes)
	triplet = np.concatenate([im1, im2, pred_color], axis=1)
	legend = create_legend_panel(class_names=class_names, num_classes=num_classes)
	legend = cv2.resize(legend, (legend.shape[1], triplet.shape[0]), interpolation=cv2.INTER_NEAREST)
	final = np.concatenate([triplet, legend], axis=1)
	cv2.imwrite(out_path, final)


def overlay_single_class(image_bgr, mask, class_id, alpha=0.55, num_classes=7):
	"""Return a copy of *image_bgr* with only *class_id* pixels highlighted."""
	palette = get_palette(num_classes)
	single_mask = np.zeros_like(mask, dtype=np.uint8)
	single_mask[mask == class_id] = 1

	color_layer = np.zeros_like(image_bgr)
	color_layer[single_mask == 1] = palette[class_id]

	output = image_bgr.copy()
	output[single_mask == 1] = cv2.addWeighted(
		image_bgr, 1 - alpha, color_layer, alpha, 0
	)[single_mask == 1]
	return output


def save_classwise_overlay(
	im1,
	im2,
	pred_mask,
	out_path,
	num_classes=7,
	class_names=None,
	alpha=0.55,
):
	"""
	Create a matplotlib figure with one subplot per *detected* semantic class
	(class_id > 0).  Each subplot shows Image 2 overlaid with only that
	single class's mask so the viewer can immediately associate a panel with
	one change type.

	Row 0 shows Image 1, Image 2, and the full semantic colour map as
	reference panels.

	Parameters
	----------
	im1, im2 : np.ndarray  (H, W, 3) BGR
	pred_mask : np.ndarray  (H, W) uint8, values 0..num_classes-1
	out_path  : str | Path
	"""
	if class_names is None:
		class_names = list(config.CLASS_NAMES)

	palette_bgr = get_palette(num_classes)

	# Classes present in prediction (skip non_change = 0)
	present = [
		cid for cid in range(1, num_classes)
		if int((pred_mask == cid).sum()) > 0
	]
	if not present:
		present = [0]  # fallback

	n_class = len(present)
	n_cols  = max(3, n_class)
	n_rows  = 1 + math.ceil(n_class / n_cols)

	fig, axes = plt.subplots(
		n_rows, n_cols,
		figsize=(5 * n_cols, 5 * n_rows),
		constrained_layout=True,
	)

	# Always index as 2-D array
	if n_rows == 1:
		axes = axes[np.newaxis, :]
	if n_cols == 1:
		axes = axes[:, np.newaxis]

	# ---- Reference row (row 0) ------------------------------------------------
	axes[0, 0].imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
	axes[0, 0].set_title("Image 1 (Before)", fontsize=12, fontweight="bold")
	axes[0, 0].axis("off")

	axes[0, 1].imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
	axes[0, 1].set_title("Image 2 (After)", fontsize=12, fontweight="bold")
	axes[0, 1].axis("off")

	semantic_color = colorize_mask(pred_mask, num_classes)
	axes[0, 2].imshow(cv2.cvtColor(semantic_color, cv2.COLOR_BGR2RGB))
	axes[0, 2].set_title("Full Semantic Prediction", fontsize=12, fontweight="bold")
	axes[0, 2].axis("off")

	for col in range(3, n_cols):
		axes[0, col].axis("off")

	# ---- Per-class rows -------------------------------------------------------
	for idx, cid in enumerate(present):
		row = 1 + idx // n_cols
		col = idx % n_cols

		overlay = overlay_single_class(im2, pred_mask, cid, alpha=alpha, num_classes=num_classes)
		axes[row, col].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

		name     = class_names[cid] if cid < len(class_names) else str(cid)
		px_count = int((pred_mask == cid).sum())
		pct      = px_count / max(1, pred_mask.size) * 100

		r, g, b = palette_bgr[cid][2], palette_bgr[cid][1], palette_bgr[cid][0]
		patch = mpatches.Patch(color=(r / 255, g / 255, b / 255), label=f"class {cid}")
		axes[row, col].legend(handles=[patch], loc="lower right", fontsize=8, framealpha=0.7)
		axes[row, col].set_title(
			f"{name}\n{px_count:,} px  ({pct:.1f}%)",
			fontsize=11,
			fontweight="bold",
		)
		axes[row, col].axis("off")

	# Hide unused cells
	for idx in range(n_class, (n_rows - 1) * n_cols):
		row = 1 + idx // n_cols
		col = idx % n_cols
		if row < n_rows and col < n_cols:
			axes[row, col].axis("off")

	fig.suptitle(
		"Per-Class Semantic Change Overlay",
		fontsize=15,
		fontweight="bold",
		y=1.01,
	)

	out_path = str(out_path)
	fig.savefig(out_path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"[classwise] Saved → {out_path}")
