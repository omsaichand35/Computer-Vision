import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import config
from models.segformer_model import SegFormerChange
from utils.dataset import ChangeDataset
from utils.visualtization import (
	overlay_mask,
	colorize_mask,
	save_prediction_triplet,
	save_prediction_with_legend,
	save_classwise_overlay,
)


def load_checkpoint(model, ckpt_path):
	try:
		state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
	except TypeError:
		state = torch.load(ckpt_path, map_location="cpu")
	if isinstance(state, dict) and "model_state" in state:
		model.load_state_dict(state["model_state"], strict=False)
	else:
		model.load_state_dict(state, strict=False)


def predict_single(model, im1_path, im2_path, img_size):
	im1 = cv2.imread(im1_path)
	im2 = cv2.imread(im2_path)
	if im1 is None or im2 is None:
		raise ValueError(f"Images not found:\n{im1_path}\n{im2_path}")

	im1 = cv2.resize(im1, (img_size, img_size))
	im2 = cv2.resize(im2, (img_size, img_size))

	img = np.concatenate([im1 / 255.0, im2 / 255.0], axis=2)
	img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0)
	img = img.to(config.DEVICE)

	with torch.no_grad():
		logits = model(img)
		logits = F.interpolate(logits, size=(img_size, img_size), mode="bilinear", align_corners=False)
		pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype("uint8")

	return im1, im2, pred


def semantic_change_stats(pred_mask):
	total_pixels = pred_mask.size
	changed = pred_mask > 0
	changed_pixels = int(changed.sum())
	changed_percent = (changed_pixels / max(1, total_pixels)) * 100.0

	class_counts = np.bincount(pred_mask.reshape(-1), minlength=config.NUM_CLASSES)

	details = []
	for class_id in range(1, min(config.NUM_CLASSES, len(class_counts))):
		count = int(class_counts[class_id])
		if count == 0:
			continue
		details.append(
			{
				"class_id": class_id,
				"class_name": config.CLASS_NAMES[class_id] if class_id < len(config.CLASS_NAMES) else str(class_id),
				"pixels": count,
				"percent_of_image": (count / max(1, total_pixels)) * 100.0,
				"percent_of_change": (count / max(1, changed_pixels)) * 100.0,
			}
		)

	return {
		"total_pixels": int(total_pixels),
		"changed_pixels": changed_pixels,
		"changed_percent": changed_percent,
		"per_class": details,
	}


def save_semantic_change_figure(im1, im2, pred, out_path):
	binary_change = (pred > 0).astype(np.uint8)
	semantic_color = colorize_mask(pred, num_classes=config.NUM_CLASSES)
	semantic_overlay = overlay_mask(im2, pred, alpha=0.45, num_classes=config.NUM_CLASSES)

	red_mask = np.zeros_like(im2)
	red_mask[binary_change == 1] = (0, 0, 255)
	binary_overlay = cv2.addWeighted(im2, 0.75, red_mask, 0.25, 0)

	fig, axes = plt.subplots(2, 3, figsize=(16, 9))

	axes[0, 0].imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
	axes[0, 0].set_title("Image 1 (Before)")
	axes[0, 0].axis("off")

	axes[0, 1].imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
	axes[0, 1].set_title("Image 2 (After)")
	axes[0, 1].axis("off")

	axes[0, 2].imshow(binary_change, cmap="gray")
	axes[0, 2].set_title("Binary Change (class > 0)")
	axes[0, 2].axis("off")

	axes[1, 0].imshow(cv2.cvtColor(semantic_color, cv2.COLOR_BGR2RGB))
	axes[1, 0].set_title("Semantic Change Map")
	axes[1, 0].axis("off")

	axes[1, 1].imshow(cv2.cvtColor(binary_overlay, cv2.COLOR_BGR2RGB))
	axes[1, 1].set_title("Binary Overlay")
	axes[1, 1].axis("off")

	axes[1, 2].imshow(cv2.cvtColor(semantic_overlay, cv2.COLOR_BGR2RGB))
	axes[1, 2].set_title("Semantic Overlay")
	axes[1, 2].axis("off")

	plt.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=180, bbox_inches="tight")
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ckpt", default=str(config.CHECKPOINT_PATH))
	parser.add_argument("--img_size", type=int, default=config.IMG_SIZE)
	parser.add_argument("--im1", default="")
	parser.add_argument("--im2", default="")
	parser.add_argument("--split", default="val", choices=["train", "val", "test"])
	parser.add_argument("--index", type=int, default=0)
	parser.add_argument("--num_images", type=int, default=30)
	parser.add_argument("--with_legend", action="store_true")
	parser.add_argument("--semantic_diff", action="store_true")
	parser.add_argument("--classwise", action="store_true", help="Plot class-wise overlays in separate grid images")
	parser.add_argument("--print_stats", action="store_true")
	parser.add_argument("--out", default="")
	args = parser.parse_args()

	default_name = "semantic_diff.png" if args.semantic_diff else "prediction.png"
	default_out = config.BASE_DIR / "outputs" / default_name
	out_path = Path(args.out) if args.out else default_out

	model = SegFormerChange(num_classes=config.NUM_CLASSES)
	load_checkpoint(model, args.ckpt)
	model.to(config.DEVICE)
	model.eval()

	if args.im1 and args.im2:
		im1, im2, pred = predict_single(model, args.im1, args.im2, args.img_size)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		if args.semantic_diff:
			save_semantic_change_figure(im1, im2, pred, out_path)
			print(f"Saved semantic change figure to: {out_path}")
		elif args.classwise:
			save_classwise_overlay(im1, im2, pred, str(out_path), num_classes=config.NUM_CLASSES, class_names=config.CLASS_NAMES)
			print(f"Saved classwise overlays grid to: {out_path}")
		elif args.with_legend:
			save_prediction_with_legend(
				im1,
				im2,
				pred,
				str(out_path),
				num_classes=config.NUM_CLASSES,
				class_names=config.CLASS_NAMES,
			)
		else:
			save_prediction_triplet(im1, im2, pred, str(out_path), num_classes=config.NUM_CLASSES)
			
		# Automatically save classwise grid alongside the main output
		if not args.classwise:
			classwise_out = out_path.parent / f"{out_path.stem}_classwise{out_path.suffix}"
			save_classwise_overlay(im1, im2, pred, str(classwise_out), num_classes=config.NUM_CLASSES, class_names=config.CLASS_NAMES)
			print(f"Saved classwise overlays grid to: {classwise_out}")
			
		print(f"Saved prediction to: {out_path}")

		if args.print_stats:
			stats = semantic_change_stats(pred)
			print(f"Changed pixels: {stats['changed_pixels']} / {stats['total_pixels']} ({stats['changed_percent']:.2f}%)")
			if not stats["per_class"]:
				print("No semantic change classes detected (only non_change).")
			else:
				print("Per-class semantic change:")
				for entry in stats["per_class"]:
					print(
						f"  - {entry['class_name']} (id={entry['class_id']}): "
						f"{entry['pixels']} px, "
						f"{entry['percent_of_image']:.2f}% of image, "
						f"{entry['percent_of_change']:.2f}% of changed area"
					)
	else:
		dataset = ChangeDataset(
			config.get_split_dir(args.split),
			img_size=args.img_size,
			return_name=True,
			num_classes=config.NUM_CLASSES,
			label_values=config.LABEL_VALUES,
		)
		count = max(1, min(args.num_images, 30))
		start_index = max(0, args.index)
		end_index = min(start_index + count, len(dataset))

		out_path.parent.mkdir(parents=True, exist_ok=True)
		base_stem = out_path.stem
		suffix = out_path.suffix if out_path.suffix else ".png"

		for idx in range(start_index, end_index):
			_, _, name = dataset[idx]
			im1_path = str(config.get_split_dir(args.split) / "im1" / name)
			im2_path = str(config.get_split_dir(args.split) / "im2" / name)
			im1, im2, pred = predict_single(model, im1_path, im2_path, args.img_size)

			multi_out = out_path.parent / f"{base_stem}_{idx:04d}{suffix}"
			if args.semantic_diff:
				save_semantic_change_figure(im1, im2, pred, multi_out)
			elif args.classwise:
				save_classwise_overlay(im1, im2, pred, str(multi_out), num_classes=config.NUM_CLASSES, class_names=config.CLASS_NAMES)
			elif args.with_legend:
				save_prediction_with_legend(
					im1,
					im2,
					pred,
					str(multi_out),
					num_classes=config.NUM_CLASSES,
					class_names=config.CLASS_NAMES,
				)
			else:
				save_prediction_triplet(im1, im2, pred, str(multi_out), num_classes=config.NUM_CLASSES)
			
			if not args.classwise:
				classwise_out = out_path.parent / f"{base_stem}_{idx:04d}_classwise{suffix}"
				save_classwise_overlay(im1, im2, pred, str(classwise_out), num_classes=config.NUM_CLASSES, class_names=config.CLASS_NAMES)
				
			print(f"Saved prediction: {multi_out}")

			if args.print_stats:
				stats = semantic_change_stats(pred)
				print(
					f"{name}: changed {stats['changed_pixels']} / {stats['total_pixels']} "
					f"({stats['changed_percent']:.2f}%)"
				)


if __name__ == "__main__":
	main()