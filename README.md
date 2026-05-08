# PikoGPT Leaderboard (Student Version)

This repository contains the public benchmark runner used to evaluate your submission via CLI inference.

The benchmark data is already preprocessed and included in this repo.
You do **not** need to run preprocessing.

## PegasusGPT Checkpoints:
Checkpoints files can be downloaded under the following links and should be placed under `Submissions/PegasusGPT_T1_run2/runs`
2. Run2 DPO UF: https://drive.google.com/file/d/1C8BXwFyR_w1mWA9-J7Qc28-r85IGy3yX/view?usp=sharing

## Local Leaderboard Run
The leaderboard was run locally with cutoff 500 using:
```bash
uv run python -m leaderboard.run_benchmarks \
  --submission PegasusGPT_T1_run2 \
  --checkpoint runs/run2dpo_best_checkpoint.pt \
  --limit 500
```


## What You Need to Provide

Create one folder per submission under `Submissions/`, for example:

```text
Submissions/
└── MyTeam/
    ├── main.py
    ├── src/
    └── runs/
        └── my_checkpoint.pt
```

Your `main.py` must support:

```bash
python main.py --stage inference \
  --checkpoint CKPT.pt \
  --prompt "..." \
  --max-tokens N \
  --temperature 0 \
  --device auto \
  --leaderboard \
  --seed 0
```

In leaderboard mode, stdout must contain only the generated completion.

## Public Benchmarks

- HellaSwag (multiple choice A-D)
- WinoGrande (binary choice A/B)
- OpenBookQA (multiple choice A-D)
- LAMBADA (next-word prediction)

## Run Evaluation

Run all public benchmarks:

```bash
uv run python -m leaderboard.run_benchmarks \
  --submission PegasusGPT_T1 \
  --checkpoint runs/pg_dpo_uf_best_checkpoint.pt \
  --limit 100
```

Run selected benchmarks only:

```bash
uv run python -m leaderboard.run_benchmarks \
  --submission PegasusGPT_T1 \
  --checkpoint runs/pg_dpo_uf_best_checkpoint.pt \
  --bench hellaswag winogrande \
  --limit 50
```

Debug run (per-example logs):

```bash
uv run python -m leaderboard.run_benchmarks \
  --submission PegasusGPT_T1 \
  --checkpoint runs/pg_dpo_uf_best_checkpoint.pt \
  --bench hellaswag \
  --limit 3 \
  --verbose
```

## Where Results Are Saved

Results are always written to:

```text
Results/<submission_name>/<checkpoint_name>/
```

Example:

```text
Results/MyTeam/my_checkpoint/
├── hellaswag/
│   └── hellaswag__my_checkpoint__validation.json
├── winogrande/
│   └── winogrande__my_checkpoint__validation.json
├── openbookqa/
│   └── openbookqa__my_checkpoint__validation.json
├── lambada/
│   └── lambada__my_checkpoint__test.json
└── MyTeam__my_checkpoint__overview.json
```

## Result File Contents

Per-benchmark JSON includes:

- benchmark name
- checkpoint path
- total / correct / invalid
- accuracy percentage
- a sample of wrong examples (`wrong_examples`) with:
  - `id`
  - `gold`
  - `pred`
  - `prompt`
  - `raw_generation`

Overview JSON includes:

- submission name
- checkpoint
- run settings (`limit`, `temperature`, `seed`, etc.)
- one summary entry per benchmark

## Build Leaderboard Table

Aggregate all runs stored under `Results/`:

```bash
uv run python leaderboard/leaderboard.py
```

Optional exports:

```bash
uv run python leaderboard/leaderboard.py --save-csv leaderboard.csv
uv run python leaderboard/leaderboard.py --save-json leaderboard.json
```
