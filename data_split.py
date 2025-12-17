import os
import shutil
from tqdm import tqdm

# 原始路径
src_image_dir = "dataset/Fashion/images"
src_mask_dir  = "dataset/Fashion/masks"

# 目标路径
dst_image_dir = "dataset/Fashion_test/images"
dst_mask_dir  = "dataset/Fashion_test/masks"

os.makedirs(dst_image_dir, exist_ok=True)
os.makedirs(dst_mask_dir, exist_ok=True)
s = 0
# 遍历 images 文件夹
for fname in tqdm(os.listdir(src_image_dir)):
    s += 1
    if s==1000:
        break
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    # 判断文件名最后部分是否是 '_full'
    base_name = os.path.splitext(fname)[0]  # 去掉后缀
    if base_name.split("_")[-1] == "front":
        # 原图路径
        src_img_path = os.path.join(src_image_dir, fname)
        src_mask_path = os.path.join(src_mask_dir, fname)  # 假设 mask 同名

        # 目标路径
        dst_img_path = os.path.join(dst_image_dir, fname)
        dst_mask_path = os.path.join(dst_mask_dir, fname)

        # 复制文件
        shutil.copy2(src_img_path, dst_img_path)
        if os.path.exists(src_mask_path):
            shutil.copy2(src_mask_path, dst_mask_path)
        else:
            print(f"Warning: mask not found for {fname}")
