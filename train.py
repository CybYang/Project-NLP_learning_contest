import os
import gc
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import logging
from torch.cuda import amp
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- Logger ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ---------------- Config ----------------
class Config:
    # 本地预训练模型路径（需下载到 pretrained/）
    model_paths = {
        "roberta": "pretrained/chinese-roberta-wwm-ext-large",
        "macbert": "pretrained/chinese-macbert-large",
        "electra": "pretrained/chinese-electra-large-discriminator",
    }

    # 数据路径（使用合并脚本生成的固定文件）
    train_file = "data/train.tsv"
    dev_file = "data/dev.tsv"      
    test_file = "data/test.tsv"    

    # 训练参数
    max_length = 64
    batch_size = 128
    accumulation_steps = 8
    epochs = 25
    weight_decay = 0.05
    warmup_ratio = 0.15
    grad_clip = 1.0
    early_stopping_patience = 3

    # 学习率
    learning_rates = [6.5e-5, 3e-5, 1e-5]

    # 数据增强
    use_symmetry_augmentation = True
    use_hard_negative_mining = True
    hard_negative_threshold = 0.4

    # 特征工程
    use_jaccard_feature = True
    use_length_diff_feature = True

    # 输出
    model_dir = "model"
    result_dir = "result"

    def __init__(self):
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)


# ---------------- FGM ----------------
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ---------------- Focal Loss ----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce)
        fl = self.alpha * ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return fl.mean()
        if self.reduction == "sum":
            return fl.sum()
        return fl


# ---------------- Dataset ----------------
class QueryPairDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128, is_train=True, data=None, labels=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if data is not None and labels is not None:
            self.data = data
            self.labels = labels
            return

        self.data, self.labels = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if is_train:
                    if len(parts) < 3:
                        continue
                    q1, q2, label = parts[0], parts[1], int(parts[2])
                    self.data.append((q1, q2))
                    self.labels.append(label)
                    if Config.use_symmetry_augmentation and label == 1:
                        self.data.append((q2, q1))
                        self.labels.append(label)
                else:
                    if len(parts) < 2:
                        continue
                    q1, q2 = parts[0], parts[1]
                    self.data.append((q1, q2))
                    self.labels.append(0)

        logger.info(f"Loaded {len(self.data)} samples from {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q1, q2 = self.data[idx]
        label = self.labels[idx]

        enc = self.tokenizer.encode_plus(
            q1, q2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        jaccard, length_diff = 0.0, 0.0
        if Config.use_jaccard_feature:
            s1, s2 = set(q1), set(q2)
            inter, uni = len(s1 & s2), len(s1 | s2)
            jaccard = inter / uni if uni > 0 else 0.0
        if Config.use_length_diff_feature:
            length_diff = abs(len(q1) - len(q2))

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "jaccard": torch.tensor(jaccard, dtype=torch.float),
            "length_diff": torch.tensor(length_diff, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ---------------- Model w/ extra features ----------------
class EnhancedClassifier(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size + 2, 2)

    def forward(self, input_ids, attention_mask, jaccard, length_diff):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        pooled = outputs.hidden_states[-1][:, 0, :]
        pooled = self.dropout(pooled)
        extra = torch.stack([jaccard, length_diff], dim=1)
        logits = self.classifier(torch.cat([pooled, extra], dim=1))
        return logits


# ---------------- Eval ----------------
def evaluate_model(model, dataloader, device, criterion=None):
    model.eval()
    all_labels, all_probs = [], []
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            jaccard = batch["jaccard"].to(device)
            length_diff = batch["length_diff"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, jaccard, length_diff)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

    auc = roc_auc_score(all_labels, all_probs)
    preds = (np.array(all_probs) > 0.5).astype(int)
    f1 = f1_score(all_labels, preds)
    acc = accuracy_score(all_labels, preds)

    result = {"auc": auc, "f1": f1, "acc": acc}
    if criterion is not None and total_samples > 0:
        result["loss"] = total_loss / total_samples
    return result, all_probs


# ---------------- Hard negative mining ----------------
def mine_hard_negatives(model, dataloader, device, threshold=0.3):
    model.eval()
    hard = []
    offset = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Mining hard negatives"):
            bs = batch["input_ids"].size(0)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            jaccard = batch["jaccard"].to(device)
            length_diff = batch["length_diff"].to(device)
            labels = batch["label"].cpu().numpy()

            logits = model(input_ids, attention_mask, jaccard, length_diff)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            for i, prob in enumerate(probs):
                if threshold <= prob <= (1 - threshold):
                    hard.append(
                        (dataloader.dataset.data[offset + i], dataloader.dataset.labels[offset + i])
                    )
            offset += bs
    return hard


# ---------------- Train one model ----------------
def train_model(model_name):
    cfg = Config()

    # 本地加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_paths[model_name], local_files_only=True, use_fast=True
    )
    train_ds = QueryPairDataset(cfg.train_file, tokenizer, cfg.max_length, is_train=True)
    dev_ds   = QueryPairDataset(cfg.dev_file,   tokenizer, cfg.max_length, is_train=True)

    logger.info(f"Training samples: {len(train_ds)} | Dev samples: {len(dev_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True
    )

    # 预训练 backbone
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_paths[model_name], num_labels=2, return_dict=True, local_files_only=True
    )
    hidden_size = base_model.config.hidden_size
    model = EnhancedClassifier(base_model, hidden_size).to(device)

    # 对抗训练
    fgm = FGM(model)

    # 损失
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    # 学习率三阶段
    lrs = cfg.learning_rates
    lr_idx = 0

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lrs[lr_idx],
        weight_decay=cfg.weight_decay
    )

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = amp.GradScaler()

    best_auc, patience = 0.0, 0
    history = []
    lr_best_auc = {lr: 0.0 for lr in lrs}

    # 备份
    original_data = train_ds.data.copy()
    original_labels = train_ds.labels.copy()

    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        cur_lr = lrs[lr_idx]

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            jaccard = batch["jaccard"].to(device)
            length_diff = batch["length_diff"].to(device)
            labels = batch["label"].to(device)

            with amp.autocast():
                logits = model(input_ids, attention_mask, jaccard, length_diff)
                loss = criterion(logits, labels) / cfg.accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * cfg.accumulation_steps

            # FGM
            fgm.attack(epsilon=0.2)
            with amp.autocast():
                logits_adv = model(input_ids, attention_mask, jaccard, length_diff)
                loss_adv = criterion(logits_adv, labels) / cfg.accumulation_steps
            scaler.scale(loss_adv).backward()
            fgm.restore()

            if (step + 1) % cfg.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        # 难样本处理
        if cfg.use_hard_negative_mining and (epoch % 2 == 1):
            logger.info(f"Mining hard negatives at epoch {epoch+1}")
            hard = mine_hard_negatives(model, train_loader, device, cfg.hard_negative_threshold)
            if hard:
                new_data = original_data + [s[0] for s in hard]
                new_labels = original_labels + [s[1] for s in hard]
                train_ds = QueryPairDataset(
                    cfg.train_file, tokenizer, cfg.max_length, is_train=True, data=new_data, labels=new_labels
                )
                train_loader = DataLoader(
                    train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
                )
                logger.info(f"Added {len(hard)} hard samples.")

        # 验证
        dev_res, _ = evaluate_model(model, dev_loader, device, criterion)
        val_auc = dev_res["auc"]
        val_loss = dev_res.get("loss", 0.0)

        logger.info(
            f"Epoch {epoch+1}/{cfg.epochs} | Time: {time.time()-t0:.2f}s | "
            f"LR: {cur_lr:.2e} | TrainLoss: {total_loss/len(train_loader):.4f} | "
            f"ValLoss: {val_loss:.4f} | ValAUC: {val_auc:.6f} | "
            f"ValF1: {dev_res['f1']:.4f} | ValAcc: {dev_res['acc']:.4f}"
        )

        lr_best_auc[cur_lr] = max(lr_best_auc[cur_lr], val_auc)

        # 学习率阶段切换
        if epoch > 2 and val_auc < best_auc and lr_idx < len(lrs) - 1:
            lr_idx += 1
            new_lr = lrs[lr_idx]
            for g in optimizer.param_groups:
                g["lr"] = new_lr
            logger.info(f"Switch learning rate -> {new_lr}")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(train_loader),
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_f1": dev_res["f1"],
                "val_acc": dev_res["acc"],
                "lr": cur_lr,
                "time": time.time() - t0,
            }
        )

        # 早停与保存
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            save_path = os.path.join("model", f"{model_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best ({best_auc:.6f}) saved to {save_path}")
        else:
            patience += 1
            logger.info(f"No improve {patience}/{cfg.early_stopping_patience}")
            if patience >= cfg.early_stopping_patience:
                logger.info(f"Early stop at epoch {epoch+1}")
                break

    # 保存训练曲线
    with open(os.path.join("model", f"{model_name}_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return best_auc, history


# ---------------- main ----------------
def main():
    cfg = Config()
    results = {}
    for name in cfg.model_paths.keys():
        logger.info(f"===> Training {name}")
        best_auc, hist = train_model(name)
        results[name] = {"best_auc": best_auc, "history": hist}
        torch.cuda.empty_cache()
        gc.collect()

    with open(os.path.join(cfg.result_dir, "training_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
