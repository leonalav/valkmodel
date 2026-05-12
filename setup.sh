apt update
apt install wget -y
apt install git -y
apt install tmux -y
apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt-get install cuda-toolkit-12-8 -y
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
DEBIAN_FRONTEND=noninteractive apt install python3.12 python3.12-venv python3.12-dev -y
python3.12 -m venv myenv

source myenv/bin/activate

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.3 --no-build-isolation
pip install transformers datasets trl accelerate peft huggingface_hub sentencepiece flash-linear-attention
pip install adam-atan2-pytorch einops wandb tqdm pydantic omegaconf hydra-core trl bitsandbytes flash-linear-attention arc-agi deepspeed
echo "done!"
