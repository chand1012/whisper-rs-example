MODELS_DIR := "./models"

MODEL_URL := "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q8_0.bin"

get-model:
  mkdir -p $(MODELS_DIR)
  curl -L $(MODEL_URL) -o $(MODELS_DIR)/ggml-medium-q8_0.bin

build:
  cargo build --release

run audio_file:
  cargo run --release -- --model $(MODELS_DIR)/ggml-medium-q8_0.bin --file {{audio_file}}
