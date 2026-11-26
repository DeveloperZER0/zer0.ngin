# zer0.ngin

zer0.ngin is an inference engine for one model, Inkling 2 Mini, on one machine: an RTX 3070 with 8 GB of VRAM and a Ryzen 9 9900X with 64 GB of RAM. It decodes up to 55 tokens per second, about twice as fast as llama.cpp on the same hardware. It serves an OpenAI-compatible API and a chat page.

The host code is Zig. The GPU code is CUDA C++, which NVRTC compiles when the engine first starts.

Polish version: [README.pl.md](README.pl.md).

## Results

Measured with `bench.py` from [inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill). Each value is the median of 6 samples of 512 tokens, at temperature 0.6 and low reasoning effort. Every engine got the same prompts. The zer0.ngin column is the median of three runs. Speeds are in tokens per second.

| Test | llama.cpp b12649, 32k profile | llama.cpp b12649, MoE cache | llama.cpp b12676, MoE cache | zer0.ngin |
|---|---:|---:|---:|---:|
| Decode, short context | 23.3 | 29.1 | 28.8 | **54.5** |
| Prefill, 3.3k tokens | 222 | 303 | 490 | **859** |
| Prefill, 12.9k tokens | 220 | 313 | 565 | **777** |
| Prefill, 18.8k tokens | 222 | 317 | 570 | **676 to 714** |
| Decode after 18.8k tokens | 23.2 | 26.6 | 26.7 | **37 to 40** |
| Needles found in an 18.8k document | 3 of 3 | 3 of 3 | 3 of 3 | 3 of 3 |

At 30.5k tokens of context zer0.ngin prefills at 686 tokens per second, decodes at 35.0 and finds 3 of 3 needles.

## Correctness

The engine was checked against llama.cpp with teacher forcing on 4 prompts. The top token matches on 195 to 200 of 201 positions, and the KL divergence is 0.001 to 0.005. Two llama.cpp builds, CUDA and CPU, agree with each other on 59 to 60 of 60 positions.

Prefill in large batches and prefill one token at a time give a KL divergence of at most 0.0003.

## How it works

- Dense weights live in VRAM. These are attention, the LM head and the loop controllers.
- About 1060 of the 3072 experts live in a VRAM cache. The cache keeps the experts used most often and uploads up to 10 new ones per token.
- The Ryzen computes the experts missing from the cache with AVX-512, from weights in RAM, while the GPU works.
- One persistent CUDA kernel computes a whole token. Its 46 blocks synchronize through barriers in VRAM. This saves about 640 kernel launches per token, at 5.5 µs each under Windows.
- Prompts longer than 160 tokens are prefilled in batches of 2048 tokens. Experts stream over PCIe and run on the tensor cores through cuBLAS.
- The next turn of a conversation reuses the tokens already processed. RAM holds checkpoints of the sliding-window state.

[THEORY.md](THEORY.md) derives the ceilings for this hardware. The engine reaches about 55% of the decode ceiling.

## Lossy changes

- The KV cache of the full-attention layers is int8, the same as `-ctk q8_0` in llama.cpp. The sliding-window cache is fp16.
- Batched prefill computes in fp16 and accumulates in fp32. Decode computes in fp32.
- The NVFP4, Q5_K and Q8_0 expert weights are the exact values from the GGUF file.

## Speculative decoding

`src/draft.zig`, `src/mega_n.cu` and `src/spec.zig` implement speculative decoding with the DSpark drafter. It is lossless: verifying 2 to 4 tokens at once gives a KL divergence of at most 8e-8 against decoding one token at a time.

It is off by default because it does not help on this machine. Each extra token in a batch uses different experts, so verifying 4 tokens costs 2.8 to 3.2 times one decode step. Code generation roughly breaks even. Prose gets slower.

## Limitations

- Text only. Images still need llama.cpp with the mmproj file.
- One request at a time. Other requests wait in a queue.
- The 46 SMs and sm_86 of the RTX 3070 are hard-coded.
- The pinned experts need about 17 GB of RAM.
- Loading takes about 20 seconds, because the engine repacks 16 GB of weights.
- Decode slows as the context grows: about 45 tokens per second at 2k tokens, 42 at 12k and 30 at 31k.

## Build and run

You need Windows, an RTX 3070, Zig 0.16.0 and these NVIDIA CUDA 12 libraries, which this repository does not include:

- `nvrtc64_120_0.dll` and `nvrtc-builtins64_*.dll`
- `cublas64_12.dll` and `cublasLt64_12.dll`

1. Run `build.cmd`. It writes `bin\zer0.ngin.exe`. Pass the path to `zig.exe` as the first argument if Zig is not in `C:\src\tools`.
2. Copy the four DLLs into `bin\`. The `--blas DIR` flag can point to another folder with the cuBLAS DLLs.
3. Download `Inkling-2-Mini-UD-Q4_K_M.gguf` from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF).
4. Start the server:

   ```
   bin\zer0.ngin.exe -m Inkling-2-Mini-UD-Q4_K_M.gguf --port 8001 -c 32768
   ```

   `Start zer0.ngin.cmd` does the same with the model path of a llama.cpp install in `%LOCALAPPDATA%\Programs\llama.cpp-Inkling`.

The chat page is at http://127.0.0.1:8001/ and the API is at http://127.0.0.1:8001/v1. The model id is `inkling-2-mini`.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Engine, CUDA kernels, tokenizer, chat template, server and tests |
| `bench/` | Benchmark scripts, run configs and the results ledger |
| `PLAN.md` | Log of every experiment, in Polish |
| `THEORY.md` | Hardware measurements and performance ceilings, in Polish |
| `THIRD_PARTY_NOTICES.md` | Licenses of the llama.cpp and benchmark code this project builds on |
