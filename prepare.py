# prepare.py 功能：读取两份原始训练集  并且进行合并去重 分层切分 
#输出：两个tsv 文件（train/dev）
import os, csv, re, random
from collections import defaultdict
from math import floor


ROUND1_TSV = "data/raw/gaiic_track3_round1_train_20210228.tsv"
ROUND2_TSV = "data/raw/gaiic_track3_round2_train_20210407.tsv"

OUT_TRAIN = "data/train.tsv"
OUT_DEV   = "data/dev.tsv"

AUGMENT_POS = True

VALID_RATIO = 0.10
SEED = 42

def ensure_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到文件：{path}")

def norm_text(s: str) -> str:
    s = s.replace("\u3000", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def read_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            if not row: continue
            row = (row + ["", "", ""])[:3]
            q1, q2, y = norm_text(row[0]), norm_text(row[1]), row[2]
            if not q1 or not q2:
                continue
            try:
                y = str(int(float(y)))
            except Exception:
                continue
            rows.append((q1, q2, y))
    return rows

def undirected_dedup(rows):
    """无向去重：把 (A,B) 与 (B,A) 视为同一条；冲突时保留正样本。"""
    seen = {}
    for q1, q2, y in rows:
        a, b = (q1, q2) if q1 <= q2 else (q2, q1)
        key = (a, b)
        if key in seen:
            if seen[key] == "0" and y == "1":
                seen[key] = "1"
        else:
            seen[key] = y
    return [(a, b, y) for (a, b), y in seen.items()]

def sym_pos_augment(rows):
    out = []
    for q1, q2, y in rows:
        out.append((q1, q2, y))
        if y == "1" and q1 != q2:
            out.append((q2, q1, "1"))
    return out

def stratified_split(rows, valid_ratio=0.1, seed=42):
    byy = defaultdict(list)
    for r in rows:
        byy[r[2]].append(r)
    rnd = random.Random(seed)
    train, valid = [], []
    for y, lst in byy.items():
        rnd.shuffle(lst)
        k = floor(len(lst) * valid_ratio)
        valid.extend(lst[:k]); train.extend(lst[k:])
    rnd.shuffle(train); rnd.shuffle(valid)
    return train, valid

def write_tsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerows(rows)

def stat(name, rows):
    n = len(rows)
    p = sum(1 for *_, y in rows if y == "1")
    print(f"{name:>6}: {n:7d} | pos={p:6d} neg={n-p:6d} | pos_rate={p/max(n,1):.3f}")

def main():
    for p in (ROUND1_TSV, ROUND2_TSV):
        ensure_exists(p)

    r1 = read_tsv(ROUND1_TSV)
    r2 = read_tsv(ROUND2_TSV)
    print("读取完成：", len(r1), len(r2))

    rows = undirected_dedup(r1 + r2)
    if AUGMENT_POS:
        rows = sym_pos_augment(rows)

    train, dev = stratified_split(rows, VALID_RATIO, SEED)
    write_tsv(OUT_TRAIN, train)
    write_tsv(OUT_DEV, dev)

    stat(" ALL", rows)
    stat("TRAIN", train)
    stat("  DEV", dev)
    print(f"已生成：{OUT_TRAIN} 与 {OUT_DEV}")

if __name__ == "__main__":
    main()
