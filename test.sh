python scripts/inference.py \
--target_region "bag" \
--attribute "style" \
--plms --outdir results/new \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/origin.png \
--reference_path examples/reference/reference.png