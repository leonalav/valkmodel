Commands for GPU SSH now:

```bash
cd /root/valkmodel
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,dev]"
```

Probe run:

```bash
valkmodel-train \
  --run-preset 130m_probe \
  --output-dir /root/runs/valkmodel-130m-probe \
  --data-config "$VALK_ROOT/configs/data_mix_fast.yaml" \
  --device cuda \
  --bf16
```

Main explicit-config run:

```bash
export VALK_ROOT=/root/valkmodel
export VALK_OUT=/root/runs/valkmodel-1b-main

valkmodel-train \
  --training-config "$VALK_ROOT/configs/training_1b_main.yaml" \
  --data-config "$VALK_ROOT/configs/data_mix_fast.yaml" \
  --tokenizer-config "$VALK_ROOT/configs/tokenizer_llama3.yaml" \
  --model-config "$VALK_ROOT/configs/valkmodel_base_1b.json" \
  --output-dir "$VALK_OUT" \
  --device cuda \
  --bf16
```

Resume:

```bash
valkmodel-train \
  --run-preset 130m_probe \
  --output-dir /root/runs/valkmodel-130m-probe \
  --resume-from /root/runs/valkmodel-130m-probe/checkpoints/step_1000 \
  --device cuda \
  --bf16
```

Pretok build:

```bash
python -m data.pretok.cli build \
  --output-dir /root/data/pretok-out \
  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset fineweb_edu \
  --dataset the_stack_v2 \
  --dataset open_web_math \
  --stage 1024 \
  --stage 2048 \
  --stage 4096 \
  --stage 8192 \
  --stage 16384 \
  --stage 32768 \
  --stage 65536 \
  --stage 131072 \
  --num-workers 4 \
  --shard-size 50000
```

Pretok publish:

```bash
python -m data.pretok.cli publish \
  --output-dir /root/data/pretok-out \
  --repo-id leonidas123/valkmodel-data
```

Smoke test one dataset:

```bash
python -m data.pretok.cli build \
  --output-dir /root/data/pretok-smoke \
  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset fineweb_edu \
  --stage 1024 \
  --limit 1000 \
  --num-workers 1
```

For Windows, start with `--num-workers 0` or `1` if multiprocessing is unstable.
