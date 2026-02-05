import re

def preprocess_text(text: str) -> str:
    """Lightweight preprocessing to normalize and deobfuscate phishing-like text."""
    if text is None:
        return ""
    s = str(text)
    s = s.replace('\r', ' ').replace('\n', ' ')
    s = s.strip()
    s = s.lower()

    # remove urls and emails (keep a token marker for heuristic checks)
    s = re.sub(r'http\S+|www\S+|https\S+', ' url ', s)
    s = re.sub(r'\S+@\S+', ' email ', s)

    # remove non-alphanumeric but keep spaces
    s = re.sub(r'[^a-z0-9\s\$]', ' ', s)

    # collapse spaced-out letters (e.g. v e r i f y -> verify)
    s = re.sub(r'(?:(?:[a-z]\s+){2,})', lambda m: m.group(0).replace(' ', ''), s)

    # collapse multiple spaces
    s = re.sub(r'\s+', ' ', s).strip()

    return s
