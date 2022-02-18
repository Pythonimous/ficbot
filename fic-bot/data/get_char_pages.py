import io
import json
import logging
import os
import re
import time

import imagehash
import pandas as pd
import requests
from PIL import Image
from jikanpy import Jikan
from tqdm import tqdm

logging.basicConfig(filename='../../logs/get_pages.log',
                    format='%(asctime)s %(message)s',
                    level=logging.ERROR)

jikan = Jikan()

with open("../../config/extract.json") as extract_config:
    config = json.load(extract_config)

data_path = "../../data/raw/anime_characters.csv"
img_path = "../../data/raw/pictures"

if not os.path.isdir(img_path):
    os.mkdir(img_path)

mal_characters = pd.read_csv(data_path)
mal_characters = mal_characters.fillna('')


def beautify_bio(bio):
    bio = bio.strip()
    bio = re.sub(r"\(Source: .+?\)$", "", bio)
    bio = "\n".join(bio.splitlines())
    bio = re.sub(r'\n+', '\n', bio).strip()

    if bio == "No biography written.":
        bio = ""
    return bio


def get_image(image_link):
    image_content = requests.get(image_link, headers=config['headers']).content
    image_bytes = io.BytesIO(image_content)
    image = Image.open(image_bytes)
    image_hash = str(imagehash.average_hash(image, hash_size=12))
    image_name = f"{img_path}/{image_hash}.jpg"
    image.save(image_name)
    return image_hash


session_count = 0

current_time = time.time()

for index, row in tqdm(mal_characters.iterrows(), total=mal_characters.shape[0]):
    if not row.img_index:  # if image has not been downloaded yet
        if session_count == 0:
            print(f'\nWelcome back!\nYou have already parsed {index} / {len(mal_characters)} characters.'
                  f'\n{len(mal_characters) - index} more to go!\nGood luck!\n')

    while True:  # basic retry iteration after an exception (in this case, URLError)
        try:
            character_index = row.mal_link.split('/')[-2]
            character = jikan.character(character_index)
            mal_characters.at[index, 'eng_name'] = character['name']
            mal_characters.at[index, 'kanji_name'] = character['name_kanji']
            mal_characters.at[index, 'bio'] = beautify_bio(character['about'])

            if character['image_url'].endswith('.jpg'):
                mal_characters.at[index, 'img_link'] = character['image_url']
                mal_characters.at[index, 'img_index'] = get_image(row['img_link'])
            else:
                mal_characters.at[index, 'img_index'] = "-1"

            session_count += 1

        except Exception as E:
            logging.error(f"{row.mal_link} failed to fetch with {str(E)}")
            mal_characters.to_csv(data_path, index_label=False)
            print(f"{row.mal_link} failed to fetch with {str(E)}")
            session_count += 1

        finally:
            if session_count % 60 == 0 and session_count != 0:
                mal_characters.to_csv(data_path, index_label=False)
                elapsed_time = current_time - time.time()
                if elapsed_time < 60:
                    time.sleep(60 - elapsed_time)  # we can only have 60 requests per minute
                current_time = time.time()

        break

mal_characters.to_csv(data_path, index_label=False)
