## WinterPear-3000
[Stella2211/DeepFloyd_IF_VRAM12GB.py](https://gist.github.com/Stella2211/ab17625d63aa03e38d82ddc8c1aae151) 간단한 변형입니다.
## How to 
### install
1. install CUDA Toolkit 11.7
2. Register CUDA Toolkit 11.7 PATH. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin
3. You must not have any other version of CUDA Toolkit in your PATH.
4. Enter the commands below line by line.
```sh
git clone https://github.com/ccvv804/WinterPear-3000
cd WinterPear-3000
python -m venv venv
.\venv\Scripts\activate
pip install deepfloyd_if==1.0.2rc0
pip install xformers
pip install git+https://github.com/openai/CLIP.git --no-deps
pip install --upgrade diffusers~=0.16 transformers~=4.28 safetensors~=0.3 sentencepiece~=0.1 accelerate~=0.18
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install https://github.com/acpopescu/bitsandbytes/releases/download/v0.38.0-win0/bitsandbytes-0.38.1-py3-none-any.whl
pip install huggingface_hub
python
from huggingface_hub import login
login()
```
5. Enter the hugging face token to access the DeepFloyd IF storage and save it.
6. Enter the commands below line by line.
```sh
exit()
```
### run
1. Enter the Python virtual environment (venv). Pass if already venv.
```sh
.\venv\Scripts\activate
```
2. Start the Python program. This program is a GUI.
```sh
python winterpear-3000.py
```
