from difflib import SequenceMatcher

def similar_ratio(original:str, value:str) -> float:
    """
    Returns the ratio of the likeness between two strings
    using SequenceMatcher.
    """
    return SequenceMatcher(None, original, value).ratio()

def similar(original:str, value:str, ratio:float=0.89) -> bool:
    """
    Returns True if strings are similar (ratio over 0.89)
    """
    return similar_ratio(original, value) >= ratio