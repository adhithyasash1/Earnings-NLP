import pytest
from datetime import datetime
import sys

sys.path.append('..')
from utils import parse_transcript_sections, calculate_readability, count_syllables


def test_parse_transcript_sections():
    text = "Prepared remarks here. Questions and Answers Q: Question? A: Answer."
    sections = parse_transcript_sections(text)

    assert 'prepared_remarks' in sections
    assert 'qa' in sections
    assert 'Prepared remarks' in sections['prepared_remarks']
    assert 'Question?' in sections['qa']


def test_calculate_readability():
    simple_text = "The cat sat on the mat."
    complex_text = "The perspicacious feline demonstrated sedentary behavior upon the textile substrate."

    simple_score = calculate_readability(simple_text)
    complex_score = calculate_readability(complex_text)

    assert simple_score > complex_score  # Simple text should have higher readability


def test_count_syllables():
    assert count_syllables("cat") == 1
    assert count_syllables("hello") == 2
    assert count_syllables("beautiful") == 4