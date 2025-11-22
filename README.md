#  NLP learning contest

基于 **RoBERTa / MacBERT / ELECTRA** 三个中文预训练模型的多模型训练与推理融合（加权平均）。

---

## 文件结构

文件结构仅作为示例，由于数据文件太大并未上传到仓库中

```
├─ data/                   # train.tsv / dev.tsv / test.tsv
├─ model/                  # 三个模型各自的最优权重 *.pth 
├─ result/                 # 逐样本概率/中间文件
├─ prediction_result/      # 最终可提交文件
├─ ensemble_result/        # 融合分析的中间导出
├─ pretrained/             # 预训练模型下载位置
├─ prepare.py              # 清洗/切分原始数据
├─ train.py                # 训练脚本
├─ predict.py              # 加载 *.pth 推理并融合，生成提交结果
└─ READEME.md              
```



---

## 整体思路

### （1）三模型集成：RoBERTa / MacBERT / ELECTRA

不同中文预训练模型对语义细节的敏感性不同（RoBERTa 更鲁棒、MacBERT 在小误差纠正和语义细节上常更强、ELECTRA 判别式预训练对难负样本更敏感）。对同一数据的**独立微调**能得到互补误差，最后**按验证集表现加权平均**概率，能降低方差、提升稳健性。 

### （2）轻量特征：在分类头拼接 “字符级 Jaccard” 与 “长度差”

纯 Transformer 能学到，但在中文短句/口语化场景，少量“显式特征”能提供**稳定的先验**。 

### （3）FGM 对抗训练（Fast Gradient Method）

文本微调容易“记模板”，对 token 噪声脆弱。**FGM**在 embedding 上做一次基于梯度的微扰，逼迫模型在局部邻域保持判别稳定性，提升**泛化与鲁棒性**。

### （4）三阶段学习率+早停

大模型微调常见“先快学粗模式，再中速稳固，最后小步抛光”。固定 LR 容易在中后期震荡；阶段式 LR 更稳。再配合**早停**防过拟合。

### （5）难样本挖掘（Hard Negative Mining）

相似检索任务中，模型容易被**边界样本**卡住。定期发现“模型最不确定”的样本（例如概率落在 0.4~0.6），让它们回流训练，**强化判别边界**。

### （6）按 dev AUC 加权融合

验证集是我们唯一可用的“近分布估计器”。按各模型在 dev 的 AUC 表现动态分配权重，常比“写死常数”更稳。



---

## 训练细节

- 在 autodl 网站租用服务器进行训练，规格： RTX 4090 * 1卡

- 单模型训练在 15h 左右，三模型共 45h 左右

- 验证集表现：

  - MacBERT: 0.968+ AUC
  - RoBERTa: 0.972+ AUC
  - ELECTRA: 0.969+ AUC

- 测试集AUC表现：0.898

  

---

## 实现步骤

### Conda 新建环境

```bash
conda create -n machine_learning python=3.10 -y
conda activate machine_learning
```

### 安装 PyTorch

安装的是GPU版，cu124轮子。

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

### 安装 modelspace

用来下载预训练模型。

```bash
pip install -U modelscope
```

### 安装其他依赖

```bash
pip install transformers==4.45.2 scikit-learn==1.5.2 pandas==2.2.3 tqdm==4.66.5 matplotlib==3.9.2
```

### 下载中文预训练模型

模型：RoBERTa-wwm-ext-large

```bash
modelscope download --model hfl/chinese-roberta-wwm-ext-large \
  --local_dir pretrained/chinese-roberta-wwm-ext-large
```

模型：MacBERT-large

```bash
modelscope download --model hfl/chinese-macbert-large \
  --local_dir pretrained/chinese-macbert-large
```

模型：ELECTRA-large

```bash
modelscope download --model hfl/chinese-electra-large-discriminator \
  --local_dir pretrained/chinese-electra-large-discriminator
```

### 可以快速验证模型是否能加载

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModel
for p in [
  "pretrained/chinese-roberta-wwm-ext-large",
  "pretrained/chinese-macbert-large",
  "pretrained/chinese-electra-large-discriminator",
]:
    AutoTokenizer.from_pretrained(p)
    AutoModel.from_pretrained(p)
    print(p, "OK")
PY
```

### 数据准备

```bash
python prepare.py
```

### 训练

```bash
python train.py
```

### 推理与融合

```bash
python predict.py
```

