# THEORY.md: sufity dla Inkling 2 Mini UD-Q4_K_M na tym komputerze

## Zmierzony sprzęt

| Zasób | Pomiar |
|---|---:|
| DDR5-6000, odczyt, 12 wątków AVX-512 | 64 GB/s |
| VRAM RTX 3070, odczyt w kernelu | ~410 GB/s |
| PCIe 4.0 x16, host do GPU | 14,1 GB/s |
| Uruchomienie kernela + flaga w pamięci hosta (WDDM) | 9,4 µs |
| Ping-pong GPU-CPU przez pamięć mapowaną, kernele w kolejce | 6,5 µs |
| Przejście między kolejkowanymi kernelami | 5,5 µs |
| GEMM fp16 x fp16 -> fp32 (cuBLAS, rdzenie tensorowe) | 10-31 TFLOPS (n = 32-1024) |

## Bajty czytane na token (48 przebiegów warstw z 24 warstw wag)

| Składnik | MB/token |
|---|---:|
| Attention Q8_0 (q, k, v, gate, out, rel), 28,5 MB x 48 | 1370 |
| Kontrolery pętli (bf16), 24 przebiegi | 113 |
| Routery (bf16) | 28 |
| Głowa LM (Q8_0) | 492 |
| Eksperci: 44 x 8 x 4,48 MB + 4 x 8 x 6,55 MB | 1790 |
| **Razem** (bez KV) | **3790** |

KV: okno przesuwne 40 x 1024 poz. x 4 KB = 168 MB/token po zapełnieniu okna; 8 pełnych przebiegów int8 = 17 KB x kontekst (553 MB przy 32k).

## Sufity dekodowania

- Wszystko z RAM: 64 / 3,79 = 17 tok/s.
- Wszystko w VRAM (model się nie mieści): 410 / 3,79 = 108 tok/s.
- Hybryda, 84% wywołań ekspertów z VRAM: część gęsta 2,0 GB / 410 = 4,9 ms, potem max(GPU 1,5 GB / 410 = 3,7 ms; CPU 0,29 GB / 64 = 4,5 ms). Razem ~9,4 ms, czyli ~106 tok/s przy idealnym nakładaniu i zerowych opóźnieniach.
- Osiągnięte 16,7-17,3 ms, czyli ok. 55% sufitu. Resztę zajmuje ok. 700 barier na token, nierówny rozkład chybień między slotami i opóźnienia krótkich etapów.

## Sufit prefillu

- Obliczenia: 5,4 mld aktywnych parametrów x 2 = 10,8 GFLOP/token; przy ~20 TFLOPS to ok. 1850 tok/s.
- PCIe: ok. 3600 brakujących ekspertów x 0,32 ms = 1,15 s na mikrobatch niezależnie od jego długości (przy 2048 tokenach: 1780 tok/s).

## Dlaczego cache per ekspert

Ślad z 4 promptów po 200 tokenów: statyczny wybór wag trafia 50-62% bajtów, cache dynamiczny 4,5 GB 80-88% (przebiegi pętli współdzielą ok. 4,5-5 z 8 ekspertów).
