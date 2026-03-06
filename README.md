# Emomni

[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/Jotakak-yu/Emomni-v1) [![models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Models-blue.svg)](https://huggingface.co/Jotakak/Emomni-v1) [![arXiv](https://img.shields.io/badge/arXiv-uploading-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/xxxx.xxxxx)

| Emomni-v1                                                             |
| --------------------------------------------------------------------- |
| 🤗 [Emomni-v1](https://huggingface.co/Jotakak/Emomni-v1) |
| 🤗 [Emomni-v1-bnb-4bit](https://huggingface.co/Jotakak/Emomni-v1-bnb-4bit) |
| 🤗 [Emomni-v1-small](https://huggingface.co/Jotakak/Emomni-v1-small) |
| 🤗 [Emomni-v1-small-bnb-4bit](https://huggingface.co/Jotakak/Emomni-v1-small-bnb-4bit) |
    
## Usage
```bash
git clone https://github.com/Jotakak-yu/Emomni-v1.git
git submodule update --init --recursive
```

- prepare tts environment
  
```bash
cd third_party/Cosyvoice
conda create -n cosy python=3.10
conda activate cosy
pip install -r requirements.txt

# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel

# download tts models
python download.py

cd pretrained_models/CosyVoice-ttsfrd/
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

- start tts server

```bash
python api.py 
```

- prepare Emomni environment
  
```bash
conda create -n emomni python=3.11
conda activate emomni
pip  install -r requirements.txt
```

- start Emomni model server

```bash
bash scripts/run_serve.sh --models path/to/model 

# optional
bash scripts/run_serve.sh --models path/to/quantized_model 

bash scripts/run_serve.sh --logs all 
bash scripts/run_serve.sh --stop
```

Access web_demo via [port 7860](http://localhost:7860)

- ps: In multi-GPU environments, it is recommended to run tts and emomni on separate GPUs to avoid resource contention.

## Acknowledgements
We would like to thank the following projects:
* [Qwen2.5](https://github.com/QwenLM/Qwen2.5)
* [blsp-emo](https://github.com/cwang621/blsp-emo)
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [cosyvoice-api](https://github.com/jianchang512/cosyvoice-api)