"""
Text Sanitization Utilities

This module provides utility functions for text sanitization in the context of natural language processing. 
Example:
    >>> from sanitize_utils import emoji2description, strip_html
    >>> text = "<p>Hello World! 😊</p>"
    >>> clean_text = strip_html(emoji2description(text))
    >>> print(clean_text)
    Hello World! face smiling with smiling eyes

Author:
    Adam Darmanin

Notes:
    Init spacy outside.
"""
from joblib import Parallel, delayed
import spacy
from spacy.tokens import Doc
from spacy.language import Language
from tqdm import tqdm
import contractions
from bs4 import BeautifulSoup
import emoji
import re
import os

DEFAULT_CHUNKS = 128

def emoji2description(text):
    """
    Converts emojis in the text to their textual description.
    
    Args:
    text (str): The input text containing emojis.
    
    Returns:
    str: Text with emojis replaced by their description.
    """
    return emoji.replace_emoji(
        text,
        replace=lambda chars, data_dict: " ".join(data_dict["en"].split("_")).strip(
            ":"
        ),
    )


def strip_html(text):
    """
    Strips HTML tags from the input text and decodes encoded symbols.
    
    Args:
    text (str): The input text with HTML tags.
    
    Returns:
    str: Text with HTML tags removed.
    """
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


@Language.component("sanitize_text")
def sanitize_text(doc):
    """
    A Spacy pipeline component that sanitizes text. It converts text to lowercase,
    expands contractions, strips HTML, and retains only specified patterns of words.
    This function is designed to be used as a part of a Spacy NLP pipeline.
    call: nlp.add_pipe("sanitize_text", first=True)
    
    Args:
    doc (Doc): A Spacy Doc object representing the input text.
    
    Returns:
    Doc: A new Spacy Doc object containing the sanitized text.
    
    Example:
        >>> import spacy
        >>> nlp.add_pipe("sanitize_text")
    """

    # Example: Convert text to lowercase
    text = doc.text.lower()

    # Expand contractions (e.g., "don't" -> "do not")
    text = contractions.fix(text)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # we want all english words and their contractions, this might include hyphenated or possisives
    words = re.findall(r"[a-z0-9\-\']{2,50}", text)

    return Doc(doc.vocab, words=words)


def clean_text_batch(texts, batch_size=DEFAULT_CHUNKS, multi_proc=False):
    """
    Cleans a batch of texts using Spacy's NLP pipeline.
    Attempts to run pipeline as multi-process on all CPUs, though has documented issues in spacy's github. See: https://github.com/explosion/spaCy/issues/5239
    see: https://spacy.io/usage/processing-pipelines#disabling for disambling unwanted pipeline components.
    
    Args:
    texts (iterable): An iterable of text strings to be cleaned.
    chunksize (int): Buffered text size for pipeline.
    multi_proc (boolean): Defaults to FALSE. TRUE if we use sPacy's native pipe multiprocessing for each CPU.
    
    Returns:
    list: A list of cleaned text strings.
    """
    nlp = spacy.load("en_core_web_sm", enable=[
                     "lemmatizer", "tagger", "parser"])
    nlp.add_pipe("sanitize_text", first=True)

    cleaned_texts = []

    # Spacy can work in token streams through pipelines, it assumes faster processing time on large corpora.
    for doc in tqdm(
        nlp.pipe(texts=texts, n_process=(os.cpu_count()-1) if multi_proc else 1, batch_size=batch_size), total=len(texts), desc="Cleaning Pipeline Token"
    ):
        cleaned_tokens = []
        for token in doc:
            if token.is_stop:
                continue
            cleaned_tokens.append(token.lemma_)
        cleaned_text = " ".join(cleaned_tokens)
        cleaned_texts.append(cleaned_text)
    return cleaned_texts


def clean_text_batch_parallel(texts, chunksize=DEFAULT_CHUNKS*10):
    """
    Multiproc version of: clean_text_batch() using pythons standard joblib.
    Chunks the text, creates tasks as much as CPUs, collects the output of each task and flattens it into 1 array.
    The tradeoff here is that we initialize the pipeline for each task.
    
    Args:
    texts (iterable): An iterable of text strings to be cleaned.
    chunksize (int): A much large chunksize than what spacy will take in the pipeline.
    
    Returns:
    list: A list of cleaned text strings.

    See: https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
    """
    def _chunker(iterable, total_length, chunksize):
        return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

    def _flatten(list_of_lists):
        "Flatten a list of lists to a combined list"
        return [item for sublist in list_of_lists for item in sublist]

    executor = Parallel(
        n_jobs=-1, backend='multiprocessing', prefer="processes")
    do = delayed(clean_text_batch)

    tasks = (do(chunk)
             for chunk in _chunker(texts, len(texts), chunksize))
    result = executor(tasks)

    return _flatten(result)

if __name__ == "__main__":
    print("Don't run")
