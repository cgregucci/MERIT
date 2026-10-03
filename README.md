# MERIT: Motif-Enriched Relational Inductive Transformer

Official implementation of **MERIT**, a knowledge graph foundation model (KGFM).

MERIT is a transformer over relation names in which the motifs enter as a soft prior
rather than as the edges of a relation graph, at two points, the attention and the
tokens. MERIT is pre-trained on link prediction and predicts links on unseen graphs
zero-shot. MERITQuery uses MERIT as the link predictor for complex query answering (CQA).

This codebase merges [ULTRA](https://arxiv.org/abs/2310.04562),
[MOTIF](https://arxiv.org/abs/2502.13339), [TRIX](https://arxiv.org/abs/2502.19512)
and [FLOCK](https://arxiv.org/abs/2510.01510), so all baselines run from the same scripts.
The evaluation per half-link scenario follows [arXiv:2606.18001](https://arxiv.org/abs/2606.18001),
and the ULTRA, MOTIF and TRIX checkpoints in `ckpts/` are the ones released with it.

## Installation

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

`requirements-frozen.txt` pins the exact environment. The CUDA kernels in `kgfm/rspmm/`
compile on first use, which needs a CUDA toolkit (`CUDA_HOME`).

FLOCK additionally needs the random-walk extension in `graph-walker/`:

```bash
CFLAGS="-Wp,-U_GLIBCXX_ASSERTIONS" CXXFLAGS="-Wp,-U_GLIBCXX_ASSERTIONS" \
    pip install -e ./graph-walker --no-build-isolation
```

## Data

Paths in the configs are relative to the repository root.

* **Link prediction.** Datasets are downloaded to `kg-datasets/` on first use.
* **CQA, Family 2 and Family 3.** The +H benchmarks of Family 2 (FB 106% to FB 550%)
  and Family 3 (WikiTopics Art, Health, Infrastructure, Location, People, Science and
  Taxonomy), valid and test, are published as a release asset. Unpack it into `cqa/`,
  so that the benchmarks sit in `cqa/benchmarks/`:

  ```bash
  curl -L -o cqa-benchmarks.zip https://github.com/cgregucci/MERIT/releases/download/benchs/benchmarks.zip
  unzip -q cqa-benchmarks.zip -d cqa && rm cqa-benchmarks.zip
  ```
* **CQA, Family 1.** FB15k237+H, NELL995+H and ICEWS18+H come from the
  [is-cqa-complex release](https://github.com/april-tools/is-cqa-complex/releases/tag/benchs-1.0):

  ```bash
  mkdir -p query-datasets && cd query-datasets
  curl -LO https://github.com/april-tools/is-cqa-complex/releases/download/benchs-1.0/iscqa-compl-benchmarks.zip
  unzip -q iscqa-compl-benchmarks.zip 'iscqa-compl-benchmarks/new_benchmarks/*'
  cd iscqa-compl-benchmarks/new_benchmarks/ICEWS18+H
  for f in train valid test; do ln -s KG_splits/$f.txt $f.txt; done
  ```

  MERITQuery is fine-tuned on the complex queries of FB15k237, which download to
  `query-datasets/`.

The +H benchmarks of Family 2 and Family 3 can be rebuilt from the original datasets.
The first step writes the queries and the second adds the 1p queries:

```bash
python cqa/script/create_queries.py --dataset InductiveFB15k237Query --version 550 \
    --root <data> --out_root <data> --splits test,valid --balanced --seed 0 --freq_cap_pct 25
python cqa/script/add_1p.py --dataroot <data>
```

Evaluation per query type and per reduction (`--test reductions`) reads reduction folders.
Build them from a +H benchmark and the held-out triples of the original dataset:

```bash
python cqa/script/build_reductions.py --dataset InductiveFB15k237Query --version 550 \
    --base <data>/550 --plus_h cqa/benchmarks/550H
```

## Usage

MERIT, link prediction:

```bash
# pre-training on FB15k237, WN18RR and CoDEx Medium (4 GPUs; -s is the pre-training seed)
python -m torch.distributed.launch --nproc_per_node=4 script/pretrain.py \
    -c config/merit/transductive/MERIT_pretrain_3g.yaml \
    --gpus [0,1,2,3] -s 2025

# zero-shot evaluation
python script/run_many.py -c config/merit/inductive/MERIT_inference.yaml \
    -d FB15k237Inductive:v1,WikiTopicsMT1:tax --gpus [0] \
    --ckpt ckpts/merit/merit_s2025.pth --test-only
python script/run_many.py -c config/merit/transductive/MERIT_inference.yaml \
    -d CoDExSmall,NELL995 --gpus [0] --ckpt ckpts/merit/merit_s2025.pth --test-only

# fine-tuning
python script/run_many.py -c config/merit/inductive/MERIT_finetune.yaml \
    -d FB15k237Inductive:v1 --gpus [0] --ckpt ckpts/merit/merit_s2025.pth \
    --finetune
```

MERITQuery, CQA (run from the repository root):

```bash
export PYTHONPATH=.:cqa

# fine-tune from a MERIT checkpoint (4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 cqa/script/run_query.py \
    -c cqa/config/meritquery/train_fb15k237.yaml --gpus [0,1,2,3] \
    --backbone_ckpt ckpts/merit/merit_s2025.pth

# evaluation on the +H benchmarks of Family 2 and Family 3
python cqa/script/run_query_many.py -c cqa/config/meritquery/inductive.yaml \
    -d InductiveFB15k237QueryH:550,WikiTopicsQueryH:tax --gpus [0] --bs 16 \
    --ckpt ckpts/meritquery/meritquery_s2025.pth -s 1024

# evaluation on the +H benchmarks of Family 1
python cqa/script/run_query_many.py -c cqa/config/meritquery/iscqa_h.yaml \
    -d FB15k237IsCqaH,NELL995IsCqaH,ICEWS18IsCqaH --gpus [0] --bs 16 \
    --ckpt ckpts/meritquery/meritquery_s2025.pth -s 1024
```

The baselines use the same scripts with their configs in `config/<model>/` and
`cqa/config/ultraquery/`.
Notes on evaluating MOTIF on AristoV4 are in `docs/`.

## Checkpoints

The checkpoints are published as a release asset. Unpack it into the repository root,
so that they sit in `ckpts/`:

```bash
curl -L -o ckpts.zip https://github.com/cgregucci/MERIT/releases/download/ckpts/ckpts.zip
unzip -q ckpts.zip && rm ckpts.zip
```

`ckpts/README.md` lists every checkpoint.

| Model | Files |
|---|---|
| MERIT | `ckpts/merit/merit_s{1024,2025,2026,4000,8000}.pth` (five pre-training runs, one per seed) |
| MERITQuery | `ckpts/meritquery/meritquery_s{1024,2025,2026,4000,8000}.pth` (fine-tuned from the MERIT pre-training run with the same seed, after the tenth epoch) |
| UltraQuery | `ckpts/ultraquery/ultraquery_3g.pth`, `ultraquery_3g_seed{2025,2026,4000,8000}.pth` (fine-tuned from the matching ULTRA pre-training run, after the tenth epoch) |
| FLOCK | `ckpts/flock/flock_entity.pth`, `flock_3g_seed{2025,2026}.pth` |
| ULTRA, MOTIF, TRIX | `ckpts/ultra/`, `ckpts/motif/`, `ckpts/trix/` |
