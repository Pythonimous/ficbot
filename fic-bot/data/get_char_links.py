import os
import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException

file_path = '../../data/raw/anime_characters.csv'

if os.path.exists(file_path):
    mal_characters = pd.read_csv(file_path)
    character_links = list(mal_characters.mal_link)
else:
    mal_characters = pd.DataFrame(columns=['eng_name',
                                           'kanji_name',
                                           'bio',
                                           'mal_link',
                                           'img_link',
                                           'img_index'])
    character_links = []

starting_letter = character_links[-1].split('/')[-1].split('_')[-1][0]
letter_links = [f"https://myanimelist.net/character.php?letter={letter}"
                for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                if letter >= starting_letter]  # resume from the latest letter

print('Resuming from letter', starting_letter)


def extract_links(source):
    soup = BeautifulSoup(source, "html.parser")
    table = soup.find('table')
    row_hrefs = [row['href'] for row in table.findAll('a', href=True)]
    char_links = []
    for href in row_hrefs:
        if href.startswith('https://myanimelist.net/character/') and href not in char_links:
            char_links.append(href)
    return char_links


driver = webdriver.Chrome('./chromedriver')  # optional argument, if not specified will search path.

for link in letter_links:

    driver.get(link)
    click_next = 2  # next page number to click

    while True:
        try:
            page_character_links = extract_links(driver.page_source)
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

mal_characters.to_csv("./anime_characters.csv", index_label=False)
