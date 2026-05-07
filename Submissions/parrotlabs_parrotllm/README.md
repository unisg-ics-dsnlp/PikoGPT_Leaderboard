# parrotlabs_parrotllm

Team **ParrotLabs** submission for the HSG NLP FS26 PikoGPT leaderboard.

## Checkpoint

The checkpoint (`parrotlabs_final.pt`, ~458 MB) exceeds GitHub's 100 MB
per-file limit and is hosted on Hugging Face at
[`ParrotLabs/parrotlabs_parrotllm`](https://huggingface.co/ParrotLabs/parrotlabs_parrotllm).

Download it into the path the runner expects:

```bash
hf download ParrotLabs/parrotlabs_parrotllm parrotlabs_final.pt \
  --local-dir Submissions/parrotlabs_parrotllm/runs
```

## Run

```bash
uv run python -m leaderboard.run_benchmarks \
  --submission parrotlabs_parrotllm \
  --checkpoint runs/parrotlabs_final.pt
```
