# ParrotLLM

A 40M-parameter decoder-only LM trained from scratch on 8B tokens of OpenWebText for the PikoGPT Challenge (NLP Lab FS26).

## Architecture

| | |
|---|---|
| Tokenizer | GPT-2 (vocab 50,257) |
| d_model | 384 |
| Layers | 14 |
| Heads | 6 (head_dim 64) |
| FFN | SwiGLU, d_ff 768 |
| Norm | RMSNorm, Pre-Norm |
| Position | RoPE |
| Tying | input embed = output head |
| Context | 1024 |
| Params | 39,966,592 |

Pretrained on `data/processed/filter_c` (filtered OpenWebText, 8B tokens), then SFT on Alpaca + a synthetic raw-format mix targeting the four leaderboard benchmarks.

## Run

The checkpoint is hosted as a release asset on the submission fork (GitHub blocks LFS uploads from public forks). Pull it once before running:

```bash
cd Submissions/ParrotLLM
bash download_checkpoint.sh    # ~150 MB into runs/
cd ../..
```

Then from the leaderboard repo root:

```bash
uv sync
uv run python -m leaderboard.run_benchmarks \
    --submission ParrotLLM \
    --checkpoint runs/final_step_0001966_epoch_01_valloss_2p4231.pt \
    --limit 500
```

The `--checkpoint` path is resolved relative to `Submissions/ParrotLLM/`, not the leaderboard root.

`main.py` implements the leaderboard contract:

```bash
python main.py --stage inference \
    --checkpoint <path> --prompt "<prompt>" \
    --max-tokens <n> --temperature 0.0 --device auto \
    --leaderboard --seed 0
```

In `--leaderboard` mode the only thing on stdout is the model's continuation: a single letter for MC prompts (HellaSwag/WinoGrande/OpenBookQA), a short word completion for LAMBADA prompts.

If `uv sync` complains about `windows-curses` not having a Python 3.14 wheel, run with Python 3.13 instead: `uv sync --python 3.13`.

## Notes on the inference path

For MC questions the model scores each option's text by length-normalized log-likelihood under the question stem (cloze) and emits the winning letter. WinoGrande's underscore-shaped stems are handled by substitution-cloze. PMI calibration (subtracting the option's loglik against a neutral "Answer:" prefix) is applied to non-substitution MC. LAMBADA prompts have their trailing space stripped before encoding so GPT-2 BPE alignment stays clean.

All four pieces live in `src/eval/inference.py`.

## Score (n=500 per benchmark)

```
hellaswag    32.2
winogrande   54.0
openbookqa   25.0
lambada      23.2
public_avg   33.6
```
