# HANDOFF Faza H — Pakiet konkursowy (po obronie 15.07.2026)

## Konkurs główny: Marian Rejewski Award (Departament Cyberbezpieczeństwa MON)

- **Deadline**: 30 listopada 2026
- **Email zgłoszenia**: konkurs.cyber@mon.gov.pl
- **Kategoria I**: prace inżynierskie, licencjackie, magisterskie
- **Nagrody**: I~--- 15 000 zł, II~--- 10 000 zł, III~--- 8 000 zł + nagrody dodatkowe
- **Kapituła**: Dyrektor Dep.~Cyberbezpieczeństwa MON, Dowódca KWOC, Szef P6 SGWP, dyrektorzy SKW/SWW, dyrektor WIŁ-PIB, dyrektor ECSC, dowódca ZDC WOT

## Wymagane dokumenty (zgodnie z prompta autora)

Lokalny CC powinien najpierw **przeczytać pełen regulamin** (dostarczony PDF 90 stron, czytać po 20):
- `a997ab8c-bRegulamin_VIII_edycji_Konkursu_o_nagrode_im._M._Rejewskiego_*.pdf`
- `9e955358-bzaaczniki_do_Regulaminu_*.docx`

Typowe wymagania (do potwierdzenia po przeczytaniu regulaminu):

1. **Pełna praca w PDF** (`thesis.pdf` z Fazy G)
2. **Formularz zgłoszeniowy** (z załączników do regulaminu)
3. **Oświadczenie autora o samodzielności pracy + zgoda na przetwarzanie danych**
4. **Zgoda promotora**
5. **Opinia promotora o pracy** (zwykle 1-2 strony A4)
6. **Streszczenie pracy** (PL + EN, do 2 stron, już w `src/str.tex`)
7. **Recenzja pracy** (opcjonalnie — np. recenzja obrony)
8. **Cover letter** (opcjonalnie ale zalecane — patrz `cover_letter_pl.tex`)

## Konkurs uzupełniający: Szef ABW

- Dokument: `c5289945-2026_06_01_regulamin_ogolnopolskiego_konkursu_Szefa_ABW.docx`
- Formularz: `9f1dd48b-2026_06_01__Zalacznik_nr_1__Formularz_zgloszenia_do_konkursu.doc`
- Termin: do potwierdzenia po przeczytaniu regulaminu
- Wymagania zwykle podobne (PDF pracy + formularz + oświadczenia)

## Krok po kroku dla lokalnego CC

### Step H1: Przeczytaj regulaminy

```bash
# Rejewski regulamin — PDF 90 str
python -c "
import sys
from pathlib import Path
# Use pdftotext jeśli zainstalowane lub odczytaj przez modale-lib
"

# ABW regulamin — DOCX
python -c "
from docx import Document
doc = Document('path/to/abw_regulamin.docx')
for para in doc.paragraphs:
    print(para.text)
"

# Wynotuj:
#  - Pełną listę wymaganych załączników
#  - Format pliku (PDF, format A4, marginesy?)
#  - Limit stron pracy (jeśli jest)
#  - Adres wysyłki
#  - Sposób potwierdzenia odbioru
#  - Czy dopuszczone są tylko prace polskojęzyczne (Rejewski przyjmuje PL + EN)
```

Zapisz notatki do `pipeline/competition_templates/regulamin_summary_rejewski.md`
i analogicznie dla ABW.

### Step H2: Wypełnij formularze zgłoszeniowe

Skopiuj formularze z załączników do regulaminu, wypełnij metadanymi:

- Tytuł pracy: ,,Porównanie metod uczenia maszynowego w wykrywaniu prokremlowskiej dezinformacji'' (PL); ,,A Comparative Study of Machine Learning Methods for Pro-Kremlin Disinformation Detection'' (EN)
- Autor: Mateusz Basaraba
- Uczelnia: Uniwersytet Marii Curie-Skłodowskiej w Lublinie (UMCS)
- Wydział: [do potwierdzenia]
- Kierunek: [do potwierdzenia]
- Promotor: prof.~Wójcik (?) [do potwierdzenia]
- Data obrony: 15.07.2026
- Słowa kluczowe: PL i EN listy z `src/str.tex`

### Step H3: Cover letter

Patrz szablon `cover_letter_pl.tex` (po PL) + `cover_letter_en.tex` (po EN). Wypełnij
konkretnymi danymi i kontrybucjami (sekcje K1, K2, K3, K4 z `scratchpad/faza-a/teza_hipotezy.md`).

### Step H4: Oświadczenia

```bash
# Szablony w pipeline/competition_templates/:
# - oświadczenie_autora.tex (samodzielność + zgoda na przetwarzanie)
# - zgoda_promotora.tex
# - opinia_promotora_template.tex (do wypełnienia przez promotora)
```

### Step H5: Build finalnego PDF

```bash
# Już mamy thesis.pdf z Fazy G. Build pakietu PDF z dodatkowymi załącznikami:
pdflatex pipeline/competition_templates/cover_letter_pl.tex
pdflatex pipeline/competition_templates/oświadczenie_autora.tex

# Pakiet:
mkdir -p submission/rejewski/
cp thesis.pdf submission/rejewski/01_thesis.pdf
cp pipeline/competition_templates/cover_letter_pl.pdf submission/rejewski/02_cover_letter.pdf
cp formularz_rejewski.pdf submission/rejewski/03_formularz_zgloszeniowy.pdf
cp oświadczenie_autora.pdf submission/rejewski/04_oswiadczenie.pdf
cp zgoda_promotora.pdf submission/rejewski/05_zgoda_promotora.pdf
cp opinia_promotora.pdf submission/rejewski/06_opinia_promotora.pdf

# Zip
cd submission/ && zip -r rejewski_submission_$(date +%Y%m%d).zip rejewski/
```

### Step H6: Wyślij email

Adres: konkurs.cyber@mon.gov.pl

Temat: ,,Zgłoszenie pracy magisterskiej do VIII edycji Konkursu im. Mariana Rejewskiego ---
Mateusz Basaraba, UMCS''

Treść: krótka, formalna; w załączniku ZIP z dokumentami.

**Przed wysłaniem**: potwierdź z autorem treść emaila i pełen pakiet.

### Step H7: Powtórz dla ABW

Analogiczny proces z dostosowaniem do regulaminu ABW.

## Pliki w tym katalogu (do uzupełnienia przez lokalnego CC)

- [ ] `regulamin_summary_rejewski.md` — wynotowane wymagania
- [ ] `regulamin_summary_abw.md` — wynotowane wymagania
- [ ] `cover_letter_pl.tex` — list motywacyjny
- [ ] `cover_letter_en.tex` — wersja angielska
- [ ] `oświadczenie_autora.tex`
- [ ] `zgoda_promotora.tex`
- [ ] `opinia_promotora_template.tex`
