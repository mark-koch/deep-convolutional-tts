"""
Utility for text normalization.
"""

import unidecode
import sys

from config import Config


def spell_out_numbers(text, languange='en'):
    """
    Spells out abbreviations and numbers.
    """
    import num2words
    # Chars, that are valid before or after numbers
    lexing_splitters = ['.', '!', '?', ',', ';', '(', ')']
    output = ""
    current_word = ""
    for c in text + " ":
        if c.isspace() or c in lexing_splitters:
            try:
                number = int(current_word)
                output += num2words.num2words(number, lang=languange) + c
            except ValueError:
                output += current_word + c
            current_word = ""
        else:
            current_word += c
    return output


def split_text(text, max_len):
    """
    Splits the text into parts to be processed by the network. Will try to split the text on sentence ends or commas.
    This will work better, if 'nltk' is installed.
    """
    nltk_installed = 'nltk' in sys.modules
    lines = text.splitlines()

    if nltk_installed:
        from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
        # Add abbreviations that are not covered by the standard nltk tokenizer of the language
        custom_abbreviations = ['mr', 'mrs']
        tokenizer = PunktSentenceTokenizer()
        for abbr in custom_abbreviations:
            tokenizer._params.abbrev_types.add(abbr)
        # Get list of sentences
        sentences = []
        for line in lines:
            if line != "" and not line.isspace():
                sentences += tokenizer.tokenize(line)
    else:
        sentences = []
        for line in lines:
            if line != "" and not line.isspace():
                sentences.append(line)

    # Find sentences that are to long and split them
    post_splitters = [',', ';', ':'] if nltk_installed else ['.', '!', '?', ',', ';', ':']
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        split_chances = []
        last_space = None
        for j in range(len(sent)):
            if sent[j] in post_splitters:
                split_chances.append(j + 1)
            if sent[j] == ' ':
                last_space = j
            if j >= max_len:  # Split needed
                if len(split_chances) > 0:
                    split = split_chances[-1]
                elif last_space is not None:
                    split = last_space
                else:
                    split = j
                a = sent[:split]
                b = sent[split:].lstrip()  # lstrip to remove space after ',' etc.
                sentences[i] = a
                sentences.insert(i + 1, b)
                break
        i += 1

    return sentences


def normalize(text):
    text = text.lower()
    # Convert to Normal Form Decomposed. We might have special letters like 'ä' in the vocab which should not be norma-
    # lized
    norm = ""
    for c in text:
        if c in Config.vocab and False:
            norm += c
        else:
            norm += unidecode.unidecode(c)
    # Lower case
    text = norm.lower()
    # Remove unknown chars
    text = "".join([char for char in text if char in Config.vocab])
    # Remove repeated whitespace and normalize tabs etc to spaces
    text = " ".join(text.split())
    return text


def vocab_lookup(text):
    vocab_char_to_idx = {char: idx for idx, char in enumerate(Config.vocab)}
    return [vocab_char_to_idx[char] for char in text]


if __name__ == "__main__":
    import os
    with open(sys.argv[1], "r") as f:
        text = f.read()
    text = spell_out_numbers(text)
    lines = split_text(text, max_len=int(sys.argv[2]))
    path = os.path.join(os.path.dirname(sys.argv[1]), "lines.txt")
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")
