# PLAN.md: dziennik eksperymentów (25.11.2026, Windows 11, plan zasilania Zrównoważony przez cały czas, kolejność chronologiczna)

Reguła przyjęcia: zgodność teacher-forced z llama.cpp (top-1 na 201 tokenach, średnie KL) oraz wynik lepszy niż kontrola.
Wyrocznia: llama.cpp (Unsloth PR 396, CPU, MinGW), zrzuty tensorów i logitów (C:\src\ref, C:\src\trace).

| # | Zmiana | Wynik | Werdykt |
|---|---|---|---|
| 1 | Baseline llama.cpp: profil 32k CUDA (z mmproj) | 23,60 / 23,58 tok/s, prefill 223 | kontrola |
| 2 | llama.cpp `--cpu-moe --moe-cache-mib 3500` | 30,09 tok/s, prefill 301 | lepsza kontrola |
| 3 | llama.cpp cache + `--moe-cache-cpu-share 1.0` | 23,28 tok/s, prefill 300 | odrzucone |
| 4 | Forward CPU w Zig (float32) | 97/100 top-1, KL 0,006, 5,8 tok/s | przyjęte jako wzorzec. Pułapka: bias względny i sink nie są skalowane przez 1/sqrt(128) |
| 5 | Silnik hybrydowy v0 (ok. 685 kerneli na token) | 197/201, KL 0,0040, 30-34 tok/s | przyjęte |
| 6 | Dostęp pasmowy (128 B na warp), scalony koniec slotu | topk 3,5 -> 1,1 ms | przyjęte |
| 7 | Warstwy 20-23 (Q5_K/Q8_0) jako zwarte płaszczyzny + 88 slotów VRAM | 38 tok/s | przyjęte |
| 8 | 10 wątków CPU zamiast 11/12 | 41 vs 39 tok/s | przyjęte |
| 9 | Wsadowe odczyty (9-12 w locie na warp) | gate_up 110 -> 72 µs | przyjęte |
| 10 | Jeden kernel na token (46 bloków, bariery w VRAM) | 52,3 tok/s | przyjęte. Po drodze: błędna pętla ctrl_up dla FFN, KL 0,68 |
| 11 | maxrregcount 128 / 80 / 64 (92-138 workerów) | 36-45 / 43 / 22-37 tok/s | odrzucone (rozlewanie rejestrów) |
| 12 | Wskaźniki w pamięci stałej | 53,6 tok/s | przyjęte |
| 13 | Polityka cache minimalizująca max(GPU, CPU) - symulacja | 1-2% | odrzucone |
| 14 | KV pełnych warstw w int8 | +155 slotów, KL 0,003-0,005 | przyjęte (zmiana stratna, oznaczona) |
| 15 | Prefill masowy (cuBLAS fp16, strumień ekspertów) | 101 -> 590 tok/s | przyjęte, KL 0,0001-0,0003 wobec ścieżki pojedynczej |
| 16 | Attention prefillu z kafelkowaniem w rejestrach | 590 -> 800 | przyjęte |
| 17 | Miksy bez dynamicznie indeksowanych tablic lokalnych | prefill 957, decode 56 | przyjęte |
| 18 | Attention dekodowania, wsad 8 z konwersją w pętli | 44 tok/s | odrzucone (255 rejestrów, rozlewanie) |
| 19 | Etapy jako funkcje noinline (wskaźniki ogólne, potem shared globalne) | 40-45 tok/s | odrzucone |
| 20 | Attention: surowe odczyty najpierw, bias osobno | 59,8 tok/s; 12k: 42,5; 31k: 30,1 | przyjęte |
| 21 | admit 4/8/16, wątki 9/10/11 na dwóch promptach | wszystko w ±1-2% | plateau |

Niezrobione dźwignie, od największej szacowanej: spekulacja DSpark (szacunek: mały zysk, bo weryfikacja k tokenów mnoży chybienia CPU),
attention przy długim kontekście rozłożone na wszystkie 46 SM, pomoc CPU w prefillu masowym, większe strony pamięci dla wag w RAM.

## STATUS (końcowe A/B, kolejka ink, llama.cpp x3, ink, ink; bench.py 6 x 512, mediany)
- zer0.ngin: c1 51,5 / 55,5 / 54,5 tok/s; prefill 3,3k 849-888, 12,9k 743-784; 18,8k: prefill 676-714, decode 37-40, igły 3/3.
- llama.cpp b12649 (Twój profil, bez mmproj): c1 23,3, prefill 222/220, 18,8k decode 23,2, igły 3/3.
- llama.cpp b12649 + moe-cache 3500: c1 29,1, prefill 303/313, 18,8k decode 26,6, igły 3/3.
- llama.cpp b12676 + moe-cache 3500: c1 28,8, prefill 490/565, 18,8k decode 26,7, igły 3/3.
- Zainstalowany CUDA-build vs wyrocznia CPU (teacher-forced): 59/60, 60/60, 60/60, 59/60.
- Decode na poziomie ok. 55% sufitu z THEORY.md. Ograniczenia: bariery (~700 na token), nierówne chybienia między slotami, opóźnienia krótkich etapów.

## Spekulacja DSpark (26.11.2026)
| # | Zmiana | Wynik | Werdykt |
|---|---|---|---|
| 22 | Pomiar w llama.cpp b12676 (6 promptów, n_max 7 / 3) | 2,37 / 2,10 tok/krok przy T=0; kod 3,4-3,9, polski 1,4 | punkt odniesienia |
| 23 | Suma ekspertów dla m kolejnych tokenów (ślad) | 1,57x / 2,03x / 2,40x dla m = 2 / 3 / 4 | koszt weryfikacji rośnie prawie liniowo |
| 24 | Draft w silniku (src/draft.cu, draft.zig), cechy z przebiegów 5/23/41 + norma końcowa | na tekście llama.cpp: 3,90 / 3,41 tok/krok (llama.cpp 3,84 / 3,38) | zgodny. Błąd po drodze: dot_q8 liczył tylko 3072 kolumny (draft ffn_down ma 9216) |
| 25 | Numeracja cech po slotach cache zamiast po przebiegach | 1,87 vs 2,84 | odrzucone |
| 26 | Kernel weryfikacji 2-4 tokenów (mega_n.cu) + hist_fix | verify vs forward: KL <= 8e-8 (2 prompty, m = 2..4, także po commit(1)); greedy identyczny na 1 z 4 promptów, reszta rozjeżdża się przy bliskich top-2 | przyjęte |
| 27 | Wektory w rejestrach dla drafta (mvx), CPU liczy eksperta raz na wszystkie tokeny, skale i selekcje w shared | draft 7,8 -> 4,6 ms; verify(3) 50 -> 37 ms | przyjęte |
| 28 | Attention wsad 4, instancje na 2/3/4 tokeny, dwa wiersze na warp | 248 rejestrów, 0 B rozlewania; verify(2/3/4) ok. 30-36 / 34-41 / 47-55 ms | częściowo |
| 29 | Spekulacja adaptacyjna (k 0..3) vs silnik bez drafta | 44-51 tok/s vs 55-60 bez drafta (draft zabiera ok. 135 slotów VRAM, wstrzykiwanie ok. 1 ms/token) | **odrzucone jako domyślne**, kod zostaje (Config.draft_path) |
| 30 | dot_q8 z pętlą po fragmentach (poprawka z #24) w kernelu tokenu | regresja 59 -> 55 tok/s | wycofane (draft używa mvx) |

Wniosek: weryfikacja k tokenów kosztuje tu (ekspertów z CPU i dekodowania NVFP4 na GPU) prawie tyle, co k osobnych tokenów.
Opłacałoby się dopiero przy verify(4) ~30 ms (dziś ~50) albo z dużo lżejszym kosztem drafta w VRAM.
