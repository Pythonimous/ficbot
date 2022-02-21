import pandas as pd
import os


def img_name_data(df_path, img_path, save_path):
    mal_characters = pd.read_csv(df_path)

    mal_w_images = mal_characters[mal_characters.img_index != "-1"]  # only rows with images
    mal_img_name = mal_w_images[["eng_name", "img_index"]].reset_index(drop=True)  # only name + image rows

    mal_img_name["eng_name"] = mal_img_name["eng_name"].map(lambda x: "@" + x + "$")  # start token + name + end token
    mal_img_name["image"] = mal_img_name["img_index"].map(lambda x: x + ".jpg")
    del mal_img_name["img_index"]

    mal_img_name.to_csv(save_path, index_label=False)


def main():
    img_name_data("../../data/raw/anime_characters.csv", "../raw/images", "../../data/interim/img_name.csv")


if __name__ == "__main__":
    main()
