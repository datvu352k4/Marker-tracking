import os
import glob
import cv2
import numpy as np
from tqdm import trange
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

INPUT_DIR   = r"images"     # Thư mục ảnh gốc
LABELS_DIR  = r"labels"  # None = label cạnh ảnh; hoặc đặt path: r"p:\Labeling Data - Marker\labels"
OUTPUT_DIR  = r"aug_out"    # Thư mục output
TARGET_COUNT = 500   # Tổng số ảnh muốn sinh
PREFIX       = "aug_" 
SEARCH_RECURSIVE = True
RANDOM_SEED  = 42

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# --- Occlusion tự chế ---
class RandomOcclusion(ImageOnlyTransform):
    def __init__(self, num_shapes=(1, 4), size_range=(0.1, 0.35), opacity_range=(0.6, 1.0), p=0.5):
        super().__init__(p=p)
        self.num_shapes = num_shapes
        self.size_range = size_range
        self.opacity_range = opacity_range

    def apply(self, img, **params):
        h, w = img.shape[:2]
        smin = int(min(h, w) * self.size_range[0])
        smax = int(min(h, w) * self.size_range[1])
        n = np.random.randint(self.num_shapes[0], self.num_shapes[1] + 1)
        out = img.copy()
        for _ in range(n):
            sh = np.random.randint(smin, smax + 1)
            sw = np.random.randint(smin, smax + 1)
            x1 = np.random.randint(0, max(1, w - sw))
            y1 = np.random.randint(0, max(1, h - sh))
            x2, y2 = x1 + sw, y1 + sh
            color = (0, 0, 0) if np.random.rand() < 0.5 else (np.random.randint(30, 100),)*3
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
            alpha = np.random.uniform(self.opacity_range[0], self.opacity_range[1])
            out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
        return out

def build_photometric_augmenter():
    return A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(1, 5), p=0.6),
            A.MedianBlur(blur_limit=2, p=0.2),
            A.Blur(blur_limit=2, p=0.2),
        ], p=0.5),
        A.MotionBlur(blur_limit=(1, 5), allow_shifted=True, p=0.5),
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.1, alpha_coef=0.02, p=0.35),

        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2),
                                   contrast_limit=(-0.3, 0.3), p=0.7),
        A.OneOf([
            A.RandomGamma(gamma_limit=(20, 40), p=1.0),  
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),    
        ], p=0.4),

        A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5),
    ], p=1.0)

def list_images(input_dir, recursive=True):
    if recursive:
        files = glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)
    else:
        files = glob.glob(os.path.join(input_dir, "*"))
    return [f for f in files if os.path.isfile(f) and f.lower().endswith(VALID_EXTS)]

def find_label_for_image(img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    local_txt = os.path.join(os.path.dirname(img_path), stem + ".txt")
    if os.path.isfile(local_txt):
        return local_txt
    if LABELS_DIR:
        cand = os.path.join(LABELS_DIR, stem + ".txt")
        if os.path.isfile(cand):
            return cand
    return None

def main():
    np.random.seed(RANDOM_SEED)

    imgs = list_images(INPUT_DIR, recursive=SEARCH_RECURSIVE)
    if not imgs:
        raise FileNotFoundError(f"Không thấy ảnh trong: {INPUT_DIR}")

    img_out_dir = os.path.join(OUTPUT_DIR, "images")
    lbl_out_dir = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    augmenter = build_photometric_augmenter()

    for k in trange(TARGET_COUNT, desc="Augmenting"):
        img_path = np.random.choice(imgs)
        img = cv2.imread(img_path)
        if img is None:
            continue

        out_img = augmenter(image=img)["image"]

        stem, ext = os.path.splitext(os.path.basename(img_path))
        out_stem = f"{PREFIX}{stem}_{k+1:04d}"

        # Lưu ảnh
        out_img_path = os.path.join(img_out_dir, out_stem + ext)
        cv2.imwrite(out_img_path, out_img)

        # Lưu label (nếu có)
        label_path = find_label_for_image(img_path)
        if label_path and os.path.isfile(label_path):
            out_label_path = os.path.join(lbl_out_dir, out_stem + ".txt")
            with open(label_path, "r", encoding="utf-8") as f_in, open(out_label_path, "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())

    print(f"Done {TARGET_COUNT} in {img_out_dir}")
    print(f"Label save {lbl_out_dir}")

if __name__ == "__main__":
    main()
