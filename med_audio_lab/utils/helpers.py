import re

def slugify(text: str, fallback: str = "record") -> str:
    """
    Очищает строку от спецсимволов для безопасного использования в именах файлов.
    """
    text = re.sub(r"[^\wа-яА-ЯёЁ-]+", "_", text.strip(), flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback