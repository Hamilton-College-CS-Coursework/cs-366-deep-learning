import argparse
from pathlib import Path
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms as T
from model_unetpp import LightningUnetPP
import lightning as L
from datamodule_oxpet import OxfordPetDataModule

# python viz.py --root ~/data --split test --classes trimap --n 12

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="~/hamilton/cpsci366/data",
                   help="Data root directory (should contain oxford-iiit-pet/ subdirectory)")
    p.add_argument("--split", type=str, default="trainval",
                   choices=["trainval", "test"], help="Data split")
    p.add_argument("--classes", type=str, default="trimap",
                   choices=["trimap", "binary"],
                   help="trimap=3 classes (pet/background/border); binary=2 classes (border merged into foreground)")
    p.add_argument("--resize", type=int, default=512,
                   help="Resize to this size on the short side (images and masks resized together)")
    p.add_argument("--n", type=int, default=12, help="Number of visualization samples")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--save-dir", type=str, default="samples", help="Output directory")
    return p.parse_args()

def mask_to_classes(mask_pil: Image.Image, mode: str = "trimap"):
    """
    Official trimap values: 1=pet/foreground, 2=background, 3=border
    Returns: integer numpy array (H,W) and list of class names
    """
    m = np.array(mask_pil, dtype=np.int64)
    if mode == "trimap":
        # Map to {0,1,2} for easier coloring and computation: 0=pet, 1=background, 2=border
        m = m - 1
        class_names = ["pet", "background", "border"]
    else:
        # binary: merge border into foreground
        pet = (m == 1) | (m == 3)
        m = pet.astype(np.int64)  # 1=pet, 0=background
        class_names = ["background", "pet"]
    return m, class_names

def colorize_mask(mask: np.ndarray, class_names):
    """Map integer label mask to RGB for visualization (manual colors for classes)."""
    H, W = mask.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    if len(class_names) == 3:
        # 0=pet(red), 1=background(black), 2=border(green)
        vis[mask == 0] = (220, 20, 60)   # red
        vis[mask == 1] = (0, 0, 0)       # black
        vis[mask == 2] = (34, 139, 34)   # green,.
    else:
        # Binary: 1=pet(red), 0=background(black)
        vis[mask == 1] = (220, 20, 60)
        vis[mask == 0] = (0, 0, 0)
    return Image.fromarray(vis, mode="RGB")

def overlay(image_pil: Image.Image, mask_rgb: Image.Image, alpha=0.45):
    """Overlay using PIL's blend to avoid dtype/size issues."""
    if image_pil.size != mask_rgb.size:
        mask_rgb = mask_rgb.resize(image_pil.size, resample=Image.NEAREST)
    img_rgb = image_pil.convert("RGB")
    return Image.blend(img_rgb, mask_rgb, alpha)

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.root).expanduser()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resize images and masks the same way
    img_tf = T.Resize(args.resize, antialias=True)
    mask_tf = T.Resize(args.resize, interpolation=Image.NEAREST)

    # Load dataset
    try:
        ds = OxfordIIITPet(
            root=str(root),
            split=args.split,
            target_types="segmentation",
            transform=img_tf,
            target_transform=mask_tf,
            download=True
        )
    except Exception as e:
        print("\n[Error] Failed to load data:", e)
        print("Please confirm the directory exists:")
        print(f"  {root}/oxford-iiit-pet/images/")
        print(f"  {root}/oxford-iiit-pet/annotations/trimaps/")
        return

    print(f"Loaded Oxford-IIIT Pet: split={args.split}, size={len(ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightningUnetPP.load_from_checkpoint("best_chkpt.ckpt")

    dm = OxfordPetDataModule(
        location=str(root),
        batch_size=1,
        num_workers=4,
        class_choice=args.classes,
        resize=args.resize
    )
    dm.setup("test")

    #Get metrics here:
    trainer = L.Trainer()
    results = trainer.test(model, dataloaders=dm.test_dataloader())
    print(results)
    model.to(device).eval()

    n = min(args.n, len(ds))
    idxs = random.sample(range(len(ds)), n)

    # Five columns: (image, real mask, real overlay, pred mask, pred overlay)
    cols = 5
    rows = n
    fig_w = 16
    fig_h = max(6, rows * 2.2)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    for r, idx in enumerate(idxs):
        img_pil, mask_pil = ds[idx]  # both already resized
        mask_arr, class_names = mask_to_classes(mask_pil, mode=args.classes)
        mask_rgb = colorize_mask(mask_arr, class_names)
        over_real = overlay(img_pil, mask_rgb, alpha=0.45)

        #Check the code below (main code block for pred)
        # img_tensor = torch.tensor(np.array(img_pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = normalize(T.ToTensor()(img_pil)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            preds = torch.softmax(logits, dim=1)
            pred_mask = torch.argmax(preds, dim=1).squeeze().cpu().numpy()
        # img_tensor = T.ToTensor()(img_pil).unsqueeze(0).to(device)
        # with torch.no_grad():
        #     logits = model(img_tensor)
        #     pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        pred_mask_rgb = colorize_mask(pred_mask, class_names)  # predicted mask
        #####
        
        over_pred = overlay(img_pil, pred_mask_rgb, alpha=0.45)

        # Handle axes for 1-row case
        if rows == 1:
            ax_img, ax_true_mask, ax_true_over, ax_pred_mask, ax_pred_over = axes[0], axes[1], axes[2], axes[3], axes[4]
        else:
            ax_img, ax_true_mask, ax_true_over, ax_pred_mask, ax_pred_over = axes[r]

        ax_img.imshow(img_pil)
        ax_img.set_title("Input Image")
        ax_img.axis("off")

        ax_true_mask.imshow(mask_rgb)
        ax_true_mask.set_title("True Mask")
        ax_true_mask.axis("off")

        ax_true_over.imshow(over_real)
        ax_true_over.set_title("True Overlay")
        ax_true_over.axis("off")

        ax_pred_mask.imshow(pred_mask_rgb)
        ax_pred_mask.set_title("Predicted Mask")
        ax_pred_mask.axis("off")

        ax_pred_over.imshow(over_pred)
        ax_pred_over.set_title("Predicted Overlay")
        ax_pred_over.axis("off")

        # Also save the first few individual overlay images
        if r < 4:
            over_pred.save(save_dir / f"overlay_{args.classes}_{r}.png")

    plt.tight_layout()
    out_path = save_dir / "oxpet_true_vs_pred.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved comparison grid to: {out_path.resolve()}")
    print("Class names:", class_names)

if __name__ == "__main__":
    main()
