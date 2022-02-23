import re
from collections import defaultdict

import pandas as pd
from num2words import num2words


def replace_text_numbers(text):
    """ Replaces all numbers in a string with words """
    if not text.endswith(' '):
        text += ' '  # a crutch for end tokens, to be fixed someday
    num_from = 0
    num_to = 0
    current_number = ""
    for i in range(len(text)):
        if current_number == "0" and not text[num_from + 1] == ' ':  # edge case: 02
            text = text[:num_from + 1] + ' ' + text[num_to + 1:]
            return replace_text_numbers(text)
        elif text[i].isnumeric() or (text[i] == "." and current_number and "." not in current_number):
            if not current_number:
                num_from = i
            current_number += text[i]
            num_to = i

        elif current_number:
            mode = 'cardinal'
            if len(current_number) > 1 and current_number.endswith("."):
                current_number = current_number[:-1]
            elif text[num_to + 1: num_to + 3] in ['st', 'nd', 'rd', 'th']:  # edge case: the 7th
                mode = 'ordinal'
                text = text[: num_to] + text[num_to + 3:]

            if mode == 'ordinal' and '.' in current_number:  # edge case: the 7.5th
                current_number = current_number.split('.')
                current_number = num2words(float(current_number[0])) \
                                 + " point " \
                                 + num2words(float(current_number[1]), to=mode)
            else:
                current_number = num2words(float(current_number), to=mode)
            text = text[:num_from] + current_number + text[num_to + 1:]
            return replace_text_numbers(text)
    return text.strip()


def clear_text(text, exception_set=None):
    """
    Clears text from all non-alphanumeric characters not in exception_set.
    Transforms non-latin characters to latin alternatives.
     """
    if exception_set is None:
        exception_set = {' ', '-', '.'}

    text = replace_text_numbers(text)

    text_clean = ''
    for char in text:
        if (char in exception_set) or (char.isalnum()):
            text_clean += char
        else:
            text_clean += ' '

    # TODO: Expand character_dict, or find pre-made dictionary
    character_dict = {
        'á': 'a',
        'ä': 'ae',
        'å': 'aa',
        'è': 'e',
        'é': 'e',
        'ê': 'e',
        'ë': 'e',
        'ó': 'o',
        'ô': 'oo',
        'ö': 'oe',
        'ø': 'oe',
        'ü': 'ue',
        'œ': 'oe',
        'š': 'sh'
    }
    rare_characters_lower = list(character_dict.keys())
    for char in rare_characters_lower:
        character_dict[char.upper()] = character_dict[char].capitalize()

    text = ''

    for char in text_clean:
        if char in character_dict:
            text += character_dict[char]
        else:
            text += char

    text = re.sub(r' +', ' ', text).strip()
    text = text[0].capitalize() + text[1:]
    return text


def clear_corpus(corpus, exclude_threshold: int = 100):
    """Clears corpus texts from all infrequent non-alphanumeric characters
    (frequency below threshold). Transforms all non-latin characters to
    latin alternatives, replaces numbers with words.
    """
    char_counts = defaultdict(int)
    for text in corpus:
        for character in text:
            char_counts[character] += 1
    exception_characters = {char for char in char_counts
                            if (char_counts[char] > exclude_threshold
                                and not char.isalnum())}
    for i in range(len(corpus)):
        corpus[i] = clear_text(corpus[i], exception_characters)
    return corpus


def img_name_data(df_path, save_path):
    mal_characters = pd.read_csv(df_path)

    mal_w_images = mal_characters[mal_characters.img_index != "-1"]  # only rows with images
    mal_img_name = mal_w_images[["eng_name", "img_index"]].reset_index(drop=True)  # only name + image rows

    mal_img_name["image"] = mal_img_name["img_index"].map(lambda x: x + ".jpg")
    del mal_img_name["img_index"]

    mal_img_name["eng_name"] = clear_corpus(mal_img_name["eng_name"], 100)

    mal_img_name.to_csv(save_path, index_label=False)


def main():
    img_name_data("../../data/raw/anime_characters.csv", "../../data/interim/img_name.csv")


if __name__ == "__main__":
    main()
