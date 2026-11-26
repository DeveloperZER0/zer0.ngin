# zer0.ngin: własny silnik inferencji Inkling 2 Mini dla RTX 3070 8 GB + Ryzen 9 9900X

English version: [README.md](README.md).

Uruchomienie: `Start zer0.ngin.cmd`, potem http://127.0.0.1:8001/ (czat) albo API OpenAI pod `http://127.0.0.1:8001/v1`.
Budowanie: `build.cmd` (wystarczy Zig 0.16.0; kernele CUDA kompiluje NVRTC przy pierwszym starcie i trzyma w `bin\*.cubin`).

Do `bin\` trzeba wrzucić biblioteki NVIDIA z CUDA 12 (nie ma ich w repo): `nvrtc64_120_0.dll`, `nvrtc-builtins64_*.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`. cuBLAS można też wskazać flagą `--blas KATALOG`. Model: `Inkling-2-Mini-UD-Q4_K_M.gguf` z `hf.co/unsloth/Inkling-2-Mini-GGUF`.

## Wyniki (bench.py ze skilla, 6 próbek po 512 tokenów, T=0,6, effort low, mediana; te same prompty dla wszystkich)

| Metryka | llama.cpp, Twój profil 32k | llama.cpp b12649 + cache MoE | llama.cpp b12676 + cache MoE | **zer0.ngin** (trzy przebiegi) |
|---|---:|---:|---:|---:|
| Decode, krótki kontekst | 23,3 tok/s | 29,1 | 28,8 | **51,5 / 55,5 / 54,5** |
| Prefill 3,3k | 222 tok/s | 303 | 490 | **849 / 859 / 888** |
| Prefill 12,9k | 220 | 313 | 565 | **743 / 784 / 777** |
| Kontekst 18,8k: prefill / decode (30,5k: 686 / 35,0, igły 3/3) | 222 / 23,2 | 317 / 26,6 | 570 / 26,7 | **676-714 / 37,0-40,0** |
| Igły (3) w dokumencie 18,8k | 3/3 | 3/3 | 3/3 | **3/3** |

Poprawność: teacher-forced top-1 wobec llama.cpp 195-200/201 na 4 promptach, KL 0,001-0,005 (llama.cpp sam
różni się od siebie: build CUDA vs CPU 59-60/60). Prefill masowy vs po jednym tokenie: KL 0,0000-0,0003.

## Jak to działa

- Zig 0.16 (host, AVX-512 na CPU, serwer, tokenizer), CUDA C++ kompilowane w locie przez NVRTC, cuBLAS z instalacji llama.cpp.
- Wagi gęste (attention, głowa, kontrolery pętli) w VRAM. ~1060 najczęściej używanych ekspertów (z 3072) siedzi w cache VRAM,
  aktualizowanym w trakcie (do 10 uploadów na token). Brakujące eksperty liczy Ryzen równolegle z GPU (AVX-512, wagi w RAM).
- Cały token to jeden kernel CUDA (46 bloków, synchronizacja barierami w VRAM): bez ~640 uruchomień po 5,5 µs.
- Prefill powyżej 160 tokenów: mikrobatche 2048 tokenów, eksperty strumieniowane przez PCIe i mnożone na rdzeniach tensorowych.
- Cache promptu: kolejna tura rozmowy reużywa wcześniejsze tokeny (checkpointy stanu okna przesuwnego w RAM).

## Zmiany stratne (oznaczone)

- KV pełnych warstw w int8 (jak `-ctk q8_0` w Twoim profilu), okno przesuwne w fp16.
- Prefill masowy liczy w fp16 (wagi i aktywacje), z akumulacją fp32. Decode liczy w fp32.
- Eksperty NVFP4/Q5_K/Q8_0 to dokładnie wartości z GGUF. Kernel NVFP4 llama.cpp na CPU dodatkowo kwantyzuje aktywacje do int8.

## Ograniczenia

- Tylko tekst: obrazy (mmproj) nadal przez llama.cpp. Jeden użytkownik naraz (kolejne żądania czekają).
- Pod tę kartę: 46 SM i sm_86 są wpisane na sztywno. Wymaga ~17 GB RAM na przypięte eksperty.
- Model w RAM po starcie: ~20 s ładowania (przepakowanie 16 GB).
- Decode spada z kontekstem: ~45 tok/s przy 2k, ~42 przy 12k, ~30 przy 31k.

Pliki: `PLAN.md` (dziennik eksperymentów), `THEORY.md` (sufity), `bench/` (harness i ledger), `src/` (kod).
