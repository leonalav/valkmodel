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