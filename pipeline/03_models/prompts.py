"""Prompt templates dla LLM zero-shot i LoRA training data.

Dwa warianty per dataset:
  - short_binary: 1-tokenowa odpowiedź (TRUE/FALSE lub DISINFO/TRUSTWORTHY).
  - cot: Chain-of-thought "explain then label".

Konsystentny format używany przez wszystkie LLM runs (Llama 3.1, Qwen 2.5).
"""

from __future__ import annotations


SHORT_BINARY_TEMPLATES = {
    "liar": (
        "You are a fact-checker. Read the political statement below and decide if it is TRUE or FALSE. "
        "Respond with exactly one word: TRUE or FALSE.\n"
        'Statement: "{text}"\n'
        "Answer (TRUE or FALSE):"
    ),
    "truthseeker": (
        "You are a fact-checker. Read the tweet below and decide if the claim is TRUE or FALSE based on "
        "common knowledge. Respond with exactly one word: TRUE or FALSE.\n"
        'Tweet: "{text}"\n'
        "Answer (TRUE or FALSE):"
    ),
    "euvsdisinfo": (
        "You are a fact-checker analyzing pro-Kremlin disinformation. Read the article excerpt below. "
        "Decide if it contains DISINFORMATION (false or manipulative pro-Kremlin narratives) or TRUSTWORTHY "
        "(factual journalism). Respond with exactly one word: DISINFORMATION or TRUSTWORTHY.\n"
        'Article: "{text}"\n'
        "Answer (DISINFORMATION or TRUSTWORTHY):"
    ),
    "pl_corpus": (
        # PL variant — instrukcja po polsku, oczekiwana odpowiedź po angielsku dla spójności parsingu
        "Jesteś weryfikatorem informacji. Przeczytaj poniższy tekst po polsku i zdecyduj czy zawiera "
        "DEZINFORMACJĘ (informacje fałszywe lub manipulujące) czy treść WIARYGODNĄ. "
        "Odpowiedz dokładnie jednym słowem: DEZINFORMACJA lub WIARYGODNA.\n"
        'Tekst: "{text}"\n'
        "Odpowiedź (DEZINFORMACJA lub WIARYGODNA):"
    ),
}

COT_TEMPLATES = {
    "liar": (
        "You are a fact-checker. Analyze the political statement and decide if it is TRUE or FALSE. "
        "First, briefly explain (1-2 sentences) the key facts that support your judgment. "
        "Then output a final line 'Final answer: TRUE' or 'Final answer: FALSE'.\n"
        'Statement: "{text}"\n'
        "Reasoning:"
    ),
    "truthseeker": (
        "You are a fact-checker. Analyze the tweet and decide if the claim is TRUE or FALSE. "
        "First, briefly explain (1-2 sentences) the key facts. "
        "Then output a final line 'Final answer: TRUE' or 'Final answer: FALSE'.\n"
        'Tweet: "{text}"\n'
        "Reasoning:"
    ),
    "euvsdisinfo": (
        "You are a fact-checker analyzing pro-Kremlin disinformation. Read the article excerpt. "
        "Briefly explain (1-2 sentences) the narrative pattern. "
        "Then output 'Final answer: DISINFORMATION' or 'Final answer: TRUSTWORTHY'.\n"
        'Article: "{text}"\n'
        "Reasoning:"
    ),
    "pl_corpus": (
        "Jesteś weryfikatorem informacji. Przeanalizuj tekst po polsku. "
        "Najpierw krótko (1-2 zdania) wyjaśnij swoje rozumowanie. "
        "Następnie wypisz w nowej linii 'Final answer: DEZINFORMACJA' lub 'Final answer: WIARYGODNA'.\n"
        'Tekst: "{text}"\n'
        "Rozumowanie:"
    ),
}

# Mapowanie tokenów odpowiedzi na etykietę binarną
TOKEN_TO_LABEL = {
    "true": 0,        # LIAR/TS: TRUE → not disinfo
    "false": 1,
    "trustworthy": 0, # EU/PL: TRUSTWORTHY → not disinfo
    "disinformation": 1,
    "wiarygodna": 0,
    "dezinformacja": 1,
}


# Aliasy: polskie pod-zbiory dziedziczą prompty po odpowiedniku zadaniowym.
# pl_claims (claim-level) ~ pl_corpus (PL DEZINFORMACJA/WIARYGODNA);
# pl_articles (document-level) ~ euvsdisinfo (DISINFORMATION/TRUSTWORTHY).
_DATASET_ALIASES = {"pl_claims": "pl_corpus", "pl_articles": "euvsdisinfo"}


def build_prompt(dataset: str, variant: str, text: str, max_text_chars: int = 1500) -> str:
    """Buduje prompt do podania modelowi.

    variant: 'short'/'short_binary' → szablon 1-słowny; 'cot' → chain-of-thought.
    Normalizacja chroni przed niespójnością ('short_binary' z configu LoRA vs 'short').
    """
    text_truncated = text[:max_text_chars]
    is_cot = "cot" in (variant or "").lower()
    templates = COT_TEMPLATES if is_cot else SHORT_BINARY_TEMPLATES
    ds = _DATASET_ALIASES.get(dataset, dataset)
    if ds not in templates:
        raise ValueError(f"Unknown dataset for prompt: {dataset}")
    return templates[ds].format(text=text_truncated)


def parse_response(response: str, variant: str = "short") -> int | None:
    """Parsuje odpowiedź modelu do etykiety 0/1. Zwraca None gdy nie da się sparsować."""
    resp = response.strip().lower()

    if variant == "cot":
        # Szukamy 'Final answer:'
        marker = "final answer:"
        if marker in resp:
            resp = resp.split(marker, 1)[1].strip()

    # Pierwsza linia, pierwsze słowo
    first_line = resp.split("\n")[0]
    # Usuń interpunkcję
    import re
    first_word = re.sub(r"[^a-zA-Ząęłńóśżźć]", "", first_line.split()[0]) if first_line.split() else ""

    return TOKEN_TO_LABEL.get(first_word)
