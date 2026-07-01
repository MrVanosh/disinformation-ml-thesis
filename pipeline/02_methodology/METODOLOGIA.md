# Metodologia eksperymentów i katalog błędów do uniknięcia

Dokument referencyjny dla pracy magisterskiej *„Wykorzystanie uczenia maszynowego
do klasyfikacji treści dezinformacyjnych"* (UMCS, M. Iwaniszczuk). Stanowi:
1. fundament rozdziału 2 (Metodologia),
2. operacyjny checklist przed każdym eksperymentem Fazy E,
3. gwarancję rygoru metodologicznego (brak data leakage, poprawne balansowanie, statystyka).

---

## 1. Hipotezy badawcze

| ID | Hipoteza | Zbiór(y) testowy | Status weryfikacji |
|----|----------|------------------|--------------------|
| **H1** | Dostrojony LLM (big-LoRA) przewyższa wyspecjalizowany encoder na krótkich, redundantnych tekstach | TruthSeeker | pilotowo potwierdzona (Qwen 7B big-LoRA 0,824 > BERT 0,794) |
| **H2** | Multilingualny encoder ≥ LLM na długich wielojęzycznych artykułach przy niższym koszcie | EUvsDisinfo | pilotowo potwierdzona (mBERT 0,952 > Llama big-LoRA 0,909) |
| **H3** | Metadane behawioralne konta nie poprawiają klasyfikacji prawdziwości pojedynczej wypowiedzi | TruthSeeker | pilotowo potwierdzona (Δ F1 < 0,001) |
| **H4** | Standardowa LoRA (8 warstw, 400 it) jest niewystarczająca; głęboka (32 warstwy, 1500 it) daje +8…+42 pkt | LIAR/TS/EU | pilotowo potwierdzona |
| **H5** | HerBERT przewyższa mBERT w transferze cross-lingual na języku polskim | PL-corpus | do weryfikacji (Faza E) |

Każdy eksperyment w macierzy Fazy E musi służyć weryfikacji co najmniej jednej hipotezy
lub stanowić niezbędny baseline. **Eksperymenty bez uzasadnienia hipotezą — wykluczamy**
(oszczędność compute).

---

## 2. Katalog błędów metodologicznych (i jak je omijamy)

Klasyfikacja dezinformacji jest szczególnie podatna na subtelne błędy, bo dane pochodzą
z fact-checków, mediów społecznościowych i scrapingu — środowisk z wbudowaną redundancją
i strukturą. Poniżej każdy błąd: **opis → jak manifestuje się w detekcji dezinformacji →
jak omijamy → status w naszym pipeline**.

### 2.1. Wyciek danych (data leakage) — pięć poziomów

Najgroźniejszy błąd: informacja z testu „przecieka" do treningu, zawyżając wyniki.
Wykrywany przez `pipeline/01_data/leakage_audit.py` (5 niezależnych audytów).

**(a) Wyciek grupowy (group-level)**
- *Manifestacja:* TruthSeeker ma 134k tweetów cytujących tylko **1062 unikalne stwierdzenia
  PolitiFact**. Losowy split umieszcza ten sam statement (ze stałą etykietą) w train i test.
  Model uczy się „statement X → label Y", nie wykrywania dezinformacji. Audyt: **99,2% grup
  testowych obecnych też w treningu**.
- *Jak omijamy:* `GroupShuffleSplit` po kluczu grupy — `statement_clean` (TS), `debunk_id`
  (EU), `claim_id` (PL-claims). Żadna grupa nie pojawia się w obu zbiorach.
- *Status:* ✅ zaimplementowane; audyt przed i po split; w pracy udokumentowany spadek
  F1 (SVM TS 0,94 → 0,71) jako dowód istotności.

**(b) Wyciek temporalny (temporal leakage)**
- *Manifestacja:* EUvsDisinfo — test zawiera artykuły wcześniejsze niż najnowsze treningowe
  (audyt: 100% test < max(train)). Model „widzi przyszłość" — uczy się tematów które
  w realnym wdrożeniu byłyby nieznane.
- *Jak omijamy:* raportujemy temporal leakage w audycie; dla zbiorów z datą rozważamy
  dodatkowy split temporalny jako analizę wrażliwości. **Decyzja:** główne wyniki na grouped
  split, ale w pracy odnotowujemy że realne wdrożenie wymagałoby splitu temporalnego.
- *Status:* ✅ audytowane; omówione jako ograniczenie.

**(c) Wyciek źródłowy (source-domain leakage)**
- *Manifestacja:* EUvsDisinfo — 88% domen testowych obecnych w treningu. Model może uczyć
  się „rozpoznawania domeny" (np. `rt.com` = dezinformacja) zamiast treści. To działa
  na benchmarku, ale zawodzi gdy dezinformacja pojawi się na nowej domenie.
- *Jak omijamy:* audytujemy; w dyskusji ostrzegamy że wysoki wynik EU częściowo odzwierciedla
  rozpoznawanie źródła. Dla rygorystycznej oceny — split po domenie (analiza wrażliwości).
- *Status:* ✅ audytowane; ostrzeżenie w dyskusji.

**(d) Wyciek near-duplicate**
- *Manifestacja:* niemal identyczne teksty (przedruki, tłumaczenia, retweety cytujące) w train
  i test. MinHash (Jaccard ≥ 0,85). EU: 6,9%, TS: 2,4%.
- *Jak omijamy:* MinHash LSH wykrywa; przy >5% — deduplikacja przed splittingiem.
- *Status:* ✅ audytowane via `datasketch`.

**(e) Niespójność etykiet w grupie (label inconsistency)**
- *Manifestacja:* TS — 63,5% statementów ma tweety z różnymi etykietami (bo etykieta jest
  per-(statement, tweet): jedne tweety zgadzają się ze stwierdzeniem, inne zaprzeczają).
  To **nie błąd** — to cecha zadania, ale wymaga świadomej obsługi przy grupowaniu.
- *Jak omijamy:* dokumentujemy; przy grupowaniu po statement zachowujemy per-tweet etykiety
  (model widzi inne tweety o tym samym stwierdzeniu — realistyczne), nie wymuszamy
  majority label (bo to zniszczyłoby sygnał).
- *Status:* ✅ udokumentowane jako świadoma decyzja projektowa.

### 2.2. Błędy w definicji zadania (task design)

**(a) Klasyfikacja debunku zamiast dezinformacji**
- *Manifestacja:* fact-checki (Demagog/OKO) to **rzetelne artykuły obalające** fałsz.
  Wzięcie treści artykułu Demagog z etykietą „Fałsz" jako label=1 uczyłoby model, że
  *dziennikarstwo fact-checkerskie = dezinformacja* — dokładna odwrotność.
- *Jak omijamy:* dla PL-claims tekstem do klasyfikacji jest **`claimReviewed`**
  (oryginalna weryfikowana teza ze schema.org ClaimReview), NIE treść artykułu.
  Treść artykułu zachowujemy osobno (`debunk_text`) jako kontekst, nigdy jako wejście.
- *Status:* ✅ naprawione w `pl_demagog_scrape.py` (text = claimReviewed).

**(b) Mieszanie poziomów zadania (claim-level vs document-level)**
- *Manifestacja:* fact-check claims (krótkie tezy ~LIAR) i pełne artykuły (~EUvsDisinfo)
  to **dwa różne zadania**. Mieszanie ich w jeden korpus daje niespójny rozkład długości
  i charakterystyki, fałszując wnioski.
- *Jak omijamy:* dwa rozdzielne polskie pod-zbiory:
  - **PL-claims** (Demagog/OKO claimReviewed) — claim-level, porównywany z LIAR
  - **PL-articles** (polski podzbiór EUvsDisinfo) — document-level, porównywany z EU
  Jasno rozróżnione w pracy jako świadoma decyzja metodologiczna.
- *Status:* ✅ zatwierdzone z autorem; opisane w rozdz. 2.

### 2.3. Balansowanie klas

**(a) Balansowanie zbioru testowego — BŁĄD**
- *Manifestacja:* sztuczne zbalansowanie testu do 50/50 zafałszowuje metryki — test musi
  odzwierciedlać **realny rozkład klas**, bo na nim mierzymy wdrożeniową skuteczność.
- *Jak omijamy:* test ZAWSZE w naturalnym rozkładzie. Balansujemy tylko (opcjonalnie) trening.
- *Status:* ✅ zasada bezwzględna.

**(b) Przesunięcie kalibracji przez balansowanie treningu (calibration drift)**
- *Manifestacja:* LoRA trenowane na zbalansowanym 50/50, testowane na 73/27 (LIAR) →
  model „nauczony" produkować 50/50 → drastyczny spadek F1. Pilotowo: Qwen LIAR
  balanced 0,215 vs natural 0,474 (+26 pkt!).
- *Jak omijamy:* ablacja `lora_natural` (trening w naturalnym rozkładzie 73/27) vs
  `lora_basic` (zbalansowany). Gdzie test niezbalansowany — trenujemy w rozkładzie zbliżonym
  do docelowego LUB stosujemy rekalibrację post-hoc (temperature scaling).
- *Status:* ✅ eksperyment kontrolny w macierzy (H4 powiązane).

**(c) Stratyfikacja w splicie**
- *Jak omijamy:* `GroupShuffleSplit` zachowuje proporcje klas na poziomie grup;
  weryfikujemy rozkład klas train vs test po splicie (powinien być zbliżony).
- *Status:* ✅ weryfikowane w `grouped_split.py`.

### 2.4. Wybór metryki

- *Błąd:* accuracy na niezbalansowanym LIAR (73% TRUE) — „model" zawsze-TRUE ma 73% acc,
  praktycznie bezużyteczny.
- *Jak omijamy:* **macro F1** jako metryka główna (uśrednia F1 per klasa, karze ignorowanie
  klasy mniejszościowej). Uzupełniająco: precision/recall per klasa, AUC-ROC
  (threshold-independent), accuracy (dla porównywalności z literaturą).
- *Status:* ✅ `metrics_with_ci.py`.

### 2.5. Rygor statystyczny

**(a) Pojedynczy seed jako wniosek — BŁĄD**
- *Manifestacja:* różnica 0,79 vs 0,81 między modelami na jednym seedzie może być szumem.
- *Jak omijamy:* ≥3 seedy (wariant C: 3-5 zależnie od kosztu) + **95% bootstrap CI**
  (1000 resamples). Wnioski tylko gdy CI się nie nakładają lub test istotności potwierdza.
- *Status:* ✅ `metrics_with_ci.bootstrap_ci`, `seeded_runner.py`.

**(b) Brak testu istotności przy porównaniu par**
- *Manifestacja:* twierdzenie „model A > model B" bez testu to spekulacja.
- *Jak omijamy:* **test McNemara** (dla sparowanych predykcji na tym samym teście) —
  sprawdza czy różnica błędów jest istotna statystycznie (p < 0,05).
- *Status:* ✅ `mcnemar.py` (zweryfikowany smoke-testem).

### 2.6. Kalibracja modeli

- *Manifestacja:* LLM bywają źle skalibrowane (Qwen ZS silnie stronniczy ku FALSE).
  Wysoka pewność ≠ poprawność. Istotne dla zastosowań OSINT (analityk ufa score'owi).
- *Jak omijamy:* **ECE** (Expected Calibration Error) + reliability diagrams +
  post-hoc **temperature scaling** dla najlepszych modeli.
- *Status:* ✅ `calibration.py` (ECE + temperature_scale).

### 2.7. Reprodukowalność

- *Jak omijamy:* (1) stałe seedy {13,42,71,89,113}; (2) **manifest danych** z SHA-256
  każdego pliku (`manifest.py`); (3) zapis wersji bibliotek + commit Git
  (`reproducibility.py`); (4) wszystkie configi w wersjonowanym YAML.
- *Status:* ✅ infrastruktura gotowa.

### 2.8. Generalizacja i przesunięcie domeny (domain shift) — PEŁNY ELEMENT

- *Manifestacja:* model trenowany na LIAR może nie działać na TS — otwarte pytanie ile
  z wyniku to „wykrywanie dezinformacji" a ile „nauczenie charakterystyki konkretnego zbioru".
- *Jak omijamy (pełny eksperyment, nie opcja):* macierz **cross-dataset transfer** —
  najlepszy model każdej rodziny trenowany na zbiorze A, ewaluowany na zbiorze B (i odwrotnie).
  Porównanie wyniku in-domain (A→A) vs out-of-domain (A→B) kwantyfikuje, ile skuteczności
  to faktyczna generalizacja, a ile dopasowanie do zbioru. Pary kompatybilne formatem:
  - claim-level: LIAR ↔ PL-claims (oba krótkie tezy z werdyktem)
  - document-level: EUvsDisinfo ↔ PL-articles (oba pełne artykuły)
  - TruthSeeker traktowany osobno (specyfika social-media).
- *Metryka:* spadek macro F1 (A→A minus A→B) jako miara „transferowalności".
- *Wniosek do pracy:* mówimy o skuteczności *na danym benchmarku* i raportujemy
  transfer jako osobny wynik; unikamy uniwersalnych twierdzeń o „wykrywaniu dezinformacji".
- *Status:* ✅ pełny element macierzy Fazy E (cross-dataset variant).

### 2.9. Analiza Pareto koszt/jakość — PEŁNY ELEMENT (podpiera wnioski projektowe)

- *Manifestacja problemu:* twierdzenia ilościowe („30× tańszy", „latencja <100 ms",
  „99,5% jakości") bez pomiaru są overclaimingiem. Konkursy i recenzenci to wychwycą.
- *Jak zapewniamy rygor:* `cost_meter.py` jako context manager wokół KAŻDEGO runu mierzy:
  - `infer_ms_per_sample` — latencja inferencji (ms/próbkę),
  - `train_s` — czas treningu,
  - `peak_ram_mb` / `peak_vram_mb` — szczyt pamięci,
  - `trainable_params` — liczba uczonych parametrów (dla LoRA vs full),
  - throughput = 1000 / `infer_ms_per_sample` (próbek/s).
- *Wynik:* wykres Pareto (oś X = koszt inferencji ms/próbkę w skali log, oś Y = macro F1).
  Front Pareto pokazuje modele „nie-zdominowane" (najlepsza jakość przy danym koszcie).
  KAŻDE twierdzenie „N× tańszy" wynika z faktycznego ilorazu zmierzonych latencji.
- *Wniosek projektowy:* trójwarstwowa architektura kaskadowa (TF-IDF → mBERT/HerBERT →
  LLM+LoRA) uzasadniona empirycznym frontem Pareto, nie założeniem.
- *Status:* ✅ pełny element Fazy E; `cost_meter` wpięty w runnery.

---

## 2bis. Threat model i kontekst zastosowania (zasada formułowania)

Praca osadzona jest w kontekście zagrożenia FIMI (*Foreign Information Manipulation and
Interference*) — udokumentowanego przez EEAS~\cite{eeas-fimi-report-2023} i objętego
dyrektywą NIS2~\cite{nis2-2022}. Wybory prezydenckie w Polsce 2025 oraz kampanie
prokremlowskie (Doppelganger, Storm-1516) stanowią konkretną motywację.

**Zasada formułowania (unikanie overclaimingu):**
- ✅ FIMI/NIS2/wybory PL jako **motywacja** — zasadne, cytowane.
- ✅ Architektura kaskadowa jako **wniosek projektowy** oparty na froncie Pareto (zmierzone
  liczby) — zasadne.
- ✅ Wymagania operacyjne (latencja, throughput) — formułowane jako **postulaty projektowe**
  weryfikowane pomiarem (cost_meter), nie jako spełnione gwarancje produkcyjne.
- ⚠️ NIE twierdzimy, że praca jest „gotowym narzędziem dla CSIRT NASK/MON" — praca dostarcza
  **empirycznych podstaw projektowych**. Sformułowania złagodzone: „podstawy projektowe",
  „przesłanki dla architektury", nie „narzędzie operacyjne dla instytucji X".
- Decyzja o sile framingu obronnego — do konsultacji z promotorem (Katedra Neuroinformatyki).

---

## 3. Zasadność trenowania (dlaczego każdy eksperyment)

| Rodzina | Po co | Hipoteza | Czy trening uzasadniony |
|---------|-------|----------|-------------------------|
| Classical (TF-IDF) | tani, interpretowalny baseline; granica osiągnięć | H1,H2 (kontekst) | tak — szybki, must-have baseline |
| Encoder (BERT/mBERT/HerBERT) | wyspecjalizowany fine-tuning; H2, H5 | H2,H5 | tak — sedno porównania encoder vs LLM |
| LLM zero-shot | czy bez treningu LLM dorównuje? | H1,H2 | tak — baseline „za darmo" (inferencja) |
| LoRA basic | dowód że płytka adaptacja jest słaba | H4 | tak, ale 1 seed wystarcza (oszczędność) |
| LoRA big | kluczowy wynik — głęboka adaptacja | H1,H4 | tak — 3-5 seedów (najważniejsze) |
| LoRA natural | ablacja kalibracji klas | H4 | tak — eksperyment kontrolny |
| Ensemble (text+metadata) | czy metadane pomagają? | H3 | tak — weryfikacja H3 |

**Zasada:** nie trenujemy modelu „bo można". Każdy run odpowiada na konkretne pytanie.
Tam gdzie pilotowy wynik jest jednoznaczny (np. basic LoRA słabe) — minimalizujemy liczbę
seedów (1 zamiast 5), oszczędzając compute na kluczowych eksperymentach.

---

## 4. Operacyjny checklist przed każdym runem Fazy E

Przed uruchomieniem dowolnego eksperymentu:

- [ ] Split pochodzi z `experiments/splits_v2/` (grouped, audytowany, dla danego seeda)
- [ ] Test set w **naturalnym rozkładzie klas** (nie balansowany)
- [ ] Dla LoRA: świadoma decyzja o rozkładzie treningu (basic=balanced, natural=naturalny)
- [ ] Model z `mlx-community/*` (lokalny, nie-gated) lub potwierdzona licencja HF
- [ ] Metryki: macro F1 + per-class + AUC + bootstrap CI
- [ ] Predykcje zapisane (`preds_v2/`) do późniejszego McNemara i analizy błędów
- [ ] Wynik zapisany z metadanymi: seed, split_sha, config, czas, koszt
- [ ] Dla ZS: test samplowany do ≤2000 (reprezentatywnie) jeśli pełny > 2000 — oszczędność

---

## 5. Co już zweryfikowano (audyt initial, Faza B)

| Zbiór | Group overlap (random) | Near-dup | Label inconsist. | Wniosek |
|-------|------------------------|----------|------------------|---------|
| LIAR | 0,0% ✅ | 0,2% ✅ | 0,0% ✅ | czysty, oficjalne splity |
| TruthSeeker | **99,2%** 🚨 | 2,4% | 63,5%* | wymaga grouped split |
| EUvsDisinfo | **59,0%** 🚨 | 6,9% 🚨 | 16,9%* | wymaga grouped + dedup |

*label inconsistency dla TS/EU = cecha zadania (per-tweet / multi-source), nie błąd.

**Konkluzja:** bez grupowego splitu wyniki na TS i EU są drastycznie zawyżone.
Cały pipeline Fazy E używa wyłącznie `experiments/splits_v2/` (grouped, 5 seedów).
