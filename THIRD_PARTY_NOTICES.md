# Third-party notices

zer0.ngin is a separate implementation. Parts of it follow llama.cpp closely:

- `src/unicode_tables.zig` is generated from llama.cpp `src/unicode-data.cpp`.
- `src/tokenizer.zig` reproduces the llama.cpp pre-tokenizer behaviour, and the model and drafter code follows the llama.cpp graphs for Inkling 2 Mini and DSpark.

llama.cpp is distributed under the MIT License:

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The benchmark scripts in `bench/` (`bench.py`, `ceiling.py`) come from sybil-solutions/inference-optimisation-skill under the MIT License, see `bench/LICENSE.skill-scripts`.

Model weights (Inkling 2 Mini GGUF, DSpark drafter GGUF) and NVIDIA libraries (cuBLAS, NVRTC) are not part of this repository.
