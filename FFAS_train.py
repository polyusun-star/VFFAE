import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from fashion_dataset import FashionpediaDataset
from torchvision import transforms
from tqdm import tqdm
from clipseg import CLIPDensePredT
from module.FFAS import FFAS
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
lr = 2e-4
epochs = 20

image_dir = "fashionpedia/images/train2020/train"
ann_file = "fashionpedia/instances_attributes_train2020.json"

test_image_dir = "fashionpedia/images/val2020/test"
test_ann_file = "fashionpedia/instances_attributes_val2020.json"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
dataset = FashionpediaDataset(image_dir, ann_file, clip_tokenizer, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

val_dataset = FashionpediaDataset(test_image_dir, test_ann_file, clip_tokenizer, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


model = FFAS(version='ViT-B/16', reduce_dim=64)
model = model.to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

from sklearn.metrics import average_precision_score
import torch.nn.functional as F

def compute_mIoU(pred_mask, true_mask, threshold=0.5, eps=1e-6):
    pred_mask = (pred_mask > threshold).float()
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    miou = (intersection + eps) / (union + eps)
    return miou.item()

def evaluate(model, val_dataloader, device):
    model.eval()
    total_miou = 0.0
    total_ap = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, input_ids, attention_mask, masks in val_dataloader:
            images = images.to(device)
            masks = masks.to(device)
            # attention_mask = attention_mask.to(device)
            
            logits = model(images, input_ids)[0]
            probas = torch.sigmoid(logits)

            for i in range(images.size(0)):
                pred = probas[i].cpu().flatten()
                gt = masks[i].cpu().flatten()
                gt = (gt > 0.5).int()

                miou = compute_mIoU(pred, gt)
                total_miou += miou

                ap = average_precision_score(gt.numpy(), pred.numpy())
                total_ap += ap

                num_samples += 1

    mean_miou = total_miou / num_samples
    mean_ap = total_ap / num_samples
    return mean_miou, mean_ap

best_miou = 0.0
best_epoch = -1
save_path = "checkpoints"
os.makedirs(save_path, exist_ok=True)

epoch_losses = []

metrics_log = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (images, input_ids, attention_mask, masks) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        masks = masks.to(device)
        attention_mask = attention_mask.to(device)

        logits = model(images, input_ids)[0]
        loss = F.binary_cross_entropy_with_logits(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    miou, ap = evaluate(model, val_dataloader, device)
    print(f"[Epoch {epoch+1}] Val mIoU: {miou:.4f}, Val AP: {ap:.4f}")
    metrics_log.append({'epoch': epoch+1, 'miou': miou, 'ap': ap})
    if miou > best_miou:
        best_miou = miou
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(save_path, "ori_best_model.pth"))
        print(f"✅ Saved best model at epoch {best_epoch} with mIoU = {best_miou:.4f}")

    epoch_losses.append(total_loss / len(dataloader))

    print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}")

with open("batch_losses.txt", "w") as f:
    for i, loss in enumerate(epoch_losses):
        f.write(f"Epoch {i+1}: {loss:.6f}\n")
