cmake_minimum_required(VERSION 3.20)

project(MyLlamaApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Existing llama.cpp installation
# ============================================================

set(LLAMA_ROOT "D:/LLM/llama.cpp")
set(LLAMA_BUILD "${LLAMA_ROOT}/build")

# ============================================================
# Application
# ============================================================

add_executable(MyLlamaApp
    main.cpp
)

# ============================================================
# llama.cpp headers
# ============================================================

target_include_directories(MyLlamaApp
    PRIVATE
        "${LLAMA_ROOT}/include"
        "${LLAMA_ROOT}/ggml/include"
)

# ============================================================
# Link against the EXISTING llama.cpp libraries
# ============================================================

target_link_libraries(MyLlamaApp
    PRIVATE
        "${LLAMA_BUILD}/src/Release/llama.lib"
        "${LLAMA_BUILD}/ggml/src/Release/ggml.lib"
        "${LLAMA_BUILD}/ggml/src/Release/ggml-base.lib"
        "${LLAMA_BUILD}/ggml/src/Release/ggml-cpu.lib"
        "${LLAMA_BUILD}/ggml/src/ggml-cuda/Release/ggml-cuda.lib"

        # Windows libraries
        bcrypt
        ws2_32
)