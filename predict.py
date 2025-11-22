import os, gc, json, math, argparse, logging
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from itertools import product

# ---------------- basic logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("predict")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- config ----------------
class Config:
    model_paths = {
        "roberta": "./pretrained/chinese-roberta-wwm-ext-large",
        "macbert": "./pretrained/chinese-macbert-large",
        "electra": "./pretrained/chinese-electra-large-discriminator",
    }
    ckpt_paths = {
        "roberta": "./model/roberta_best.pth",
        "macbert": "./model/macbert_best.pth",
        "electra": "./model/electra_best.pth",
    }
    max_length = 64
    batch_size = 256
    num_workers = 2
    data_train = "./data/train.tsv"  
    data_dev   = "./data/dev.tsv"    
    data_test  = "./data/test.tsv"   
    out_dir    = "./prediction_result"  # 产出 result.tsv
    os.makedirs(out_dir, exist_ok=True)

# ---------------- tiny dataset ----------------
class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, has_label: bool):
        self.q1 = df.iloc[:, 0].astype(str).tolist()
        self.q2 = df.iloc[:, 1].astype(str).tolist()
        self.labels = df.iloc[:, 2].astype(int).tolist() if has_label and df.shape[1] >= 3 else None
        self.tok = tokenizer
        self.maxlen = max_length

    def __len__(self): return len(self.q1)

    def __getitem__(self, i):
        a, b = self.q1[i], self.q2[i]
        enc = self.tok(
            a, b, padding="max_length", truncation=True,
            max_length=self.maxlen, return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "q1": a, "q2": b
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

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
            output_hidden_states=True
        )
        pooled = outputs.hidden_states[-1][:, 0, :]
        pooled = self.dropout(pooled)
        extra = torch.stack([jaccard, length_diff], dim=1)
        logits = self.classifier(torch.cat([pooled, extra], dim=1))
        return logits

# ---------------- simple text sims ----------------
def jaccard_char(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb: return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def build_tfidf_corpus(train_dev_df: pd.DataFrame) -> TfidfVectorizer:
    vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2))
    vec.fit(pd.concat([train_dev_df.iloc[:,0], train_dev_df.iloc[:,1]]).astype(str).tolist())
    return vec

def tfidf_cosine(vec: TfidfVectorizer, a: List[str], b: List[str]) -> np.ndarray:
    A = vec.transform(a); B = vec.transform(b)
    num = A.multiply(B).sum(axis=1).A.ravel()
    den = np.maximum(1e-8, np.sqrt((A.multiply(A)).sum(axis=1).A.ravel()) * np.sqrt((B.multiply(B)).sum(axis=1).A.ravel()))
    return num / den

# ---------------- inference for one backbone with symmetric TTA ----------------
@torch.no_grad()
def infer_backbone(name: str, cfg: Config, df: pd.DataFrame, vec: TfidfVectorizer) -> Dict[str, np.ndarray]:
    log.info(f"[{name}] loading tokenizer & model...")
    tok = AutoTokenizer.from_pretrained(cfg.model_paths[name], local_files_only=True, use_fast=True)
    base = AutoModelForSequenceClassification.from_pretrained(cfg.model_paths[name], num_labels=2, return_dict=True, local_files_only=True)
    hidden = base.config.hidden_size
    model = EnhancedClassifier(base, hidden).to(device)
    sd = torch.load(cfg.ckpt_paths[name], map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    ds_fwd = PairDataset(df, tok, cfg.max_length, has_label=False)
    df_rev = df[[1,0]].copy()
    df_rev.columns = [0,1]
    ds_rev = PairDataset(df_rev, tok, cfg.max_length, has_label=False)

    dl_fwd = DataLoader(ds_fwd, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    dl_rev = DataLoader(ds_rev, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    probs_fwd, probs_rev = [], []
    jacs_fwd, lens_fwd = [], []
    jacs_rev, lens_rev = [], []

    tfidf_fwd = tfidf_cosine(vec, df.iloc[:,0].astype(str).tolist(), df.iloc[:,1].astype(str).tolist())
    tfidf_rev = tfidf_cosine(vec, df.iloc[:,1].astype(str).tolist(), df.iloc[:,0].astype(str).tolist())

    for (batch_f, batch_r, i) in zip(dl_fwd, dl_rev, range(len(dl_fwd))):
        # fwd
        q1f = batch_f["q1"]; q2f = batch_f["q2"]
        jac_f = torch.tensor([jaccard_char(a,b) for a,b in zip(q1f,q2f)], dtype=torch.float, device=device)
        len_f = torch.tensor([abs(len(a)-len(b)) for a,b in zip(q1f,q2f)], dtype=torch.float, device=device)
        logits = model(batch_f["input_ids"].to(device), batch_f["attention_mask"].to(device), jac_f, len_f)
        pf = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
        probs_fwd.append(pf)
        jacs_fwd.append(jac_f.detach().cpu().numpy())
        lens_fwd.append(len_f.detach().cpu().numpy())

        # rev
        q1r = batch_r["q1"]; q2r = batch_r["q2"]
        jac_r = torch.tensor([jaccard_char(a,b) for a,b in zip(q1r,q2r)], dtype=torch.float, device=device)
        len_r = torch.tensor([abs(len(a)-len(b)) for a,b in zip(q1r,q2r)], dtype=torch.float, device=device)
        logits = model(batch_r["input_ids"].to(device), batch_r["attention_mask"].to(device), jac_r, len_r)
        pr = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
        probs_rev.append(pr)
        jacs_rev.append(jac_r.detach().cpu().numpy())
        lens_rev.append(len_r.detach().cpu().numpy())

    p_f = np.concatenate(probs_fwd)
    p_r = np.concatenate(probs_rev)
    p_sym = (p_f + p_r) / 2.0

    jac = (np.concatenate(jacs_fwd) + np.concatenate(jacs_rev)) / 2.0
    ldf = (np.concatenate(lens_fwd) + np.concatenate(lens_rev)) / 2.0
    tfidf = (tfidf_fwd + tfidf_rev) / 2.0

    del model, base; torch.cuda.empty_cache(); gc.collect()

    return {"prob": p_sym, "jaccard": jac, "len_diff": ldf, "tfidf": tfidf}

# ---------------- ensembling helpers ----------------
def rank_normalize(x: np.ndarray) -> np.ndarray:
    r = x.argsort().argsort().astype(np.float32)
    return (r + 1) / (len(x) + 1)

def search_weights_on_dev(dev_label: np.ndarray, model_probs: Dict[str, np.ndarray],
                          sim_feat: np.ndarray) -> Tuple[Dict[str,float], float]:
    names = sorted(model_probs.keys())
    best_auc, best = -1.0, None
    grid = np.arange(0, 1.05, 0.05)
    beta_grid = np.arange(0.0, 0.22, 0.02)

    rp = {k: rank_normalize(v) for k, v in model_probs.items()}

    for w in product(grid, repeat=len(names)):
        if abs(sum(w) - 1.0) > 1e-6:  
            continue
        w = list(w)
        base = np.zeros_like(dev_label, dtype=np.float32)
        for i, k in enumerate(names):
            base += w[i] * rp[k]

        for beta in beta_grid:
            pred = (1 - beta) * base + beta * rank_normalize(sim_feat)
            auc = roc_auc_score(dev_label, pred)
            if auc > best_auc:
                best_auc = auc
                best = (w, beta)

    best_w = {k: float(best[0][i]) for i, k in enumerate(names)}
    best_beta = float(best[1])
    log.info(f"[dev] best AUC={best_auc:.6f}, weights={best_w}, beta={best_beta:.3f}")
    return best_w, best_beta

# ---------------- main ----------------
def main():
    cfg = Config()

    dev_df  = pd.read_csv(cfg.data_dev,  sep="\t", header=None)   # [q1, q2, label]
    test_df = pd.read_csv(cfg.data_test, sep="\t", header=None)   # [q1, q2]
    log.info(f"dev={len(dev_df)}, test={len(test_df)}")

    corpus_df = dev_df
    if os.path.exists(cfg.data_train):
        try:
            tr = pd.read_csv(cfg.data_train, sep="\t", header=None)
            corpus_df = pd.concat([dev_df[[0,1]], tr[[0,1]]], axis=0, ignore_index=True)
        except Exception:
            pass
    vec = build_tfidf_corpus(corpus_df)

    dev_probs, test_probs = {}, {}
    dev_sim,   test_sim   = None, None

    for name in cfg.model_paths.keys():
        log.info(f"==> infer dev with {name}")
        d = infer_backbone(name, cfg, dev_df[[0,1]], vec)
        dev_probs[name] = d["prob"]

        s_dev = 0.6*d["tfidf"] + 0.3*d["jaccard"] + 0.1*(1.0/(1.0+np.exp(d["len_diff"]/8.0)))
        dev_sim = s_dev if dev_sim is None else (dev_sim + s_dev) / 2.0  # 多模型平均一下该信号

        log.info(f"==> infer test with {name}")
        t = infer_backbone(name, cfg, test_df[[0,1]], vec)
        test_probs[name] = t["prob"]
        s_test = 0.6*t["tfidf"] + 0.3*t["jaccard"] + 0.1*(1.0/(1.0+np.exp(t["len_diff"]/8.0)))
        test_sim = s_test if test_sim is None else (test_sim + s_test) / 2.0

    y_dev = dev_df.iloc[:,2].values.astype(int)

    best_w, beta = search_weights_on_dev(y_dev, dev_probs, dev_sim)

    names = sorted(test_probs.keys())
    rp_test = {k: rank_normalize(v) for k, v in test_probs.items()}
    blend = np.zeros(len(test_df), dtype=np.float32)
    for k in names:
        blend += best_w[k] * rp_test[k]
    final = (1 - beta) * blend + beta * rank_normalize(test_sim)
    final = np.clip(final, 0.0, 1.0)

    os.makedirs(cfg.out_dir, exist_ok=True)
    sub_path = os.path.join(cfg.out_dir, "result.tsv")
    pd.DataFrame(final, columns=["score"]).to_csv(sub_path, sep="\t", header=False, index=False)
    log.info(f"✅ submission saved to {sub_path}")

    an_path = os.path.join(cfg.out_dir, "ensemble_dev_analysis.csv")
    tmp = pd.DataFrame({"y": y_dev})
    for k in dev_probs:
        tmp[f"{k}_prob"] = dev_probs[k]
        tmp[f"{k}_rank"] = rank_normalize(dev_probs[k])
    tmp["sim_feat"] = dev_sim

    rb = {k: rank_normalize(dev_probs[k]) for k in dev_probs}
    mix = np.zeros_like(y_dev, dtype=np.float32)
    for k in dev_probs: mix += best_w[k]*rb[k]
    tmp["final_rank"] = (1-beta)*mix + beta*rank_normalize(dev_sim)
    tmp["auc_final"] = roc_auc_score(y_dev, tmp["final_rank"])
    tmp.to_csv(an_path, index=False)
    log.info(f"📊 dev analysis saved to {an_path} (AUC_final={tmp['auc_final'][0]:.6f})")

if __name__ == "__main__":
    main()
