# MyLlamaApp

## C++ Local LLM Application using NVIDIA CUDA

MyLlamaApp is a native C++ command-line chatbot that runs a local Large Language Model (LLM) on NVIDIA GPU hardware using CUDA acceleration.

The application integrates:

- C++17
- llama.cpp
- CUDA
- NVIDIA GeForce MX450 GPU
- Qwen2.5-0.5B-Instruct
- GGUF / Q4_K_M quantized model
- CMake
- Visual Studio / MSVC

The application supports interactive multi-turn conversations and performs local LLM inference without requiring a cloud-based LLM API.

---

## Architecture

```text
                    MyLlamaApp
                        │
                        ▼
             ┌─────────────────────┐
             │   C++ Application   │
             │      main.cpp       │
             └──────────┬──────────┘
                        │
                        ▼
              ┌──────────────────┐
              │   llama.cpp API  │
              └────────┬─────────┘
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
        CPU Backend        CUDA Backend
                                 │
                                 ▼
                       NVIDIA GeForce MX450
                              2 GB VRAM
                                 │
                                 ▼
                  Qwen2.5-0.5B-Instruct
                       Q4_K_M GGUF