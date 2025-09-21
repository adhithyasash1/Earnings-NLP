"""Utilities"""
import logging
import re

def setup_logging(level: str = "INFO"):
    logging.basicConfig(level=getattr(logging, level), format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def parse_transcript_sections(text: str) -> dict:
    qa_match = re.search(r"Questions and Answers|Q&A Session|Question-and-Answer Session", text, re.IGNORECASE)
    if qa_match:
        return {'prepared_remarks': text[:qa_match.start()].strip(), 'qa': text[qa_match.start():].strip()}
    return {'prepared_remarks': text, 'qa': ""}

def calculate_readability(text: str) -> float:
    if not text:
        return 0.0
    sentences = len(re.findall(r'[.!?]+', text))
    words = len(text.split())
    if sentences == 0 or words == 0:
        return 0.0
    syllables = sum(count_syllables(word) for word in text.split())
    if syllables == 0:
        return 0.0
    score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    return max(0, min(100, score))

def count_syllables(word: str) -> int:
    word = word.lower().strip(".:,;?!")
    if not word:
        return 0
    vowels = "aeiouy"
    count = int(word[0] in vowels)
    for i in range(1, len(word)):
        if word[i] in vowels and word[i-1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    return max(count, 1)