import argparse
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
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from tqdm import tqdm


def beautify_bio(bio):
    bio = bio.strip()
    bio = re.sub(r"\(Source: .+?\)$", "", bio)
    bio = "\n".join(bio.splitlines())
    bio = re.sub(r'\n+', '\n', bio).strip()

    if bio == "No biography written.":
        bio = ""
    return bio


def get_image(image_link, image_path, config):
    """ Download an iamge and save it under a unique hash-name """
    image_content = requests.get(image_link, headers=config['headers']).content
    image_bytes = io.BytesIO(image_content)
    image = Image.open(image_bytes)
    image_hash = str(imagehash.average_hash(image, hash_size=12))
    image_name = f"{image_path}/{image_hash}.jpg"
    image.save(image_name)
    return image_hash


def extract_links_from_page(source):
    """ Extracts all character links from MAL html source """
    soup = BeautifulSoup(source, "html.parser")
    table = soup.find('table')
    row_hrefs = [row['href'] for row in table.findAll('a', href=True)]
    char_links = []
    for href in row_hrefs:
        if href.startswith('https://myanimelist.net/character/') and href not in char_links:
            char_links.append(href)
    return char_links


def download_links(data_path):
    """ Downloads all character links from MyAnimeList using Selenium """
    if os.path.exists(data_path):
        mal_characters = pd.read_csv(data_path)
        character_links = list(mal_characters.mal_link)
    else:
        mal_characters = pd.DataFrame(columns=['eng_name',
                                               'kanji_name',
                                               'bio',
                                               'mal_link',
                                               'img_link',
                                               'img_index'])
        character_links = []

    if character_links:
        starting_letter = character_links[-1].split('/')[-1].split('_')[-1][0]
    else:
        starting_letter = "A"

    letter_links = [f"https://myanimelist.net/character.php?letter={letter}"
                    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    if letter >= starting_letter]  # resume from the latest letter

    print('Resuming from letter', starting_letter)

    driver = webdriver.Chrome('./chromedriver')  # optional argument, if not specified will search path.

    for link in letter_links:

        driver.get(link)
        click_next = 2  # next page number to click

        while True:
            try:
                page_character_links = extract_links_from_page(driver.page_source)
                if link == letter_links[0]:
                    character_links += [link for link in page_character_links if link not in character_links]
                else:
                    character_links += page_character_links
                pages = driver.find_element_by_class_name("normal_header")
                next_page = pages.find_element_by_link_text(str(click_next))
                next_page.click()
                time.sleep(1)
                click_next += 1
            except NoSuchElementException:  # if clickable page number doesn't exist (we are at the last numbered page)
                break

    driver.quit()

    mal_characters.mal_link = character_links

    mal_characters.to_csv(data_path, index_label=False)


def download_characters(data_path, img_dir, config):
    """ Downloads all character data and images using JikanAPI for MAL """
    mal_characters = pd.read_csv(data_path).fillna('')

    session_count = 0

    current_time = time.time()

    for index, row in tqdm(mal_characters.iterrows(), total=mal_characters.shape[0]):
        if not row.img_index:  # if image has not been downloaded yet
            if session_count == 0:
                print(f'\nWelcome back!\nYou have already parsed {index} / {len(data_path)} characters.'
                      f'\n{len(mal_characters) - index} more to go!\nGood luck!\n')

        while True:  # basic retry iteration after an exception (in this case, URLError)
            try:
                character_id = row.mal_link.split('/')[-2]
                character = requests.get(f'https://api.jikan.moe/v4/characters/{character_id}').json()['data']
                mal_characters.at[index, 'eng_name'] = character['name']
                mal_characters.at[index, 'kanji_name'] = character['name_kanji']
                mal_characters.at[index, 'bio'] = beautify_bio(character['about'])

                image_url = character['images']['jpg']['image_url']
                if image_url.endswith('.jpg'):
                    mal_characters.at[index, 'img_link'] = image_url
                    mal_characters.at[index, 'img_index'] = get_image(row['img_link'], img_dir, config)
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


def main(get_links, data_path, img_dir, config_path, log_path):

    logging.basicConfig(filename=log_path,
                        format='%(asctime)s %(message)s',
                        level=logging.ERROR)

    with open(config_path) as extract_config:
        config = json.load(extract_config)

    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)

    if get_links:
        print('Extracting links first!')
        download_links(data_path)
        print('Extracting links finished! Launch the script without --get_links next.')
    else:
        print('Extracting character data!')
        download_characters(data_path, img_dir, config)
        print('All done!')


def parse_arguments():
    parser = argparse.ArgumentParser(description='This script downloads MyAnimeList data '
                                                 'from character links using v4 of Jikan API.')
    parser.add_argument('--get_links', action='store_false',
                        help='download character links first')

    parser.add_argument('--data_path', default='../data/interim/anime_characters.csv', metavar='DATA_PATH',
                        help='set csv file for saving character data')
    parser.add_argument('--img_dir', default='../data/interim/images', metavar='IMG_PATH',
                        help='set directory for saving character images')

    parser.add_argument('--config_path', default='config/extract.json', metavar='CONFIG_PATH',
                        help='set config for requests library file path')
    parser.add_argument('--log_path', default='logs/get_pages.log', metavar='LOG_PATH',
                        help='set logging file path')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)
