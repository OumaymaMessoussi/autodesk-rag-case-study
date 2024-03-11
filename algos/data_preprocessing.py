import os
import re
from glob import glob

import pandas as pd
from bs4 import BeautifulSoup

from global_variables import HTML_PAGES_FOLDER_PATH, PREPROCESSED_DATA_CSV_PATH


def convert_html_to_text(html_page_path: str) -> str:
    """This function takes a path to an html page and converts its content to string format.

    :param html_page_path: path to an html page.
    :return: a string containing the text content of the page.
    """
    with open(html_page_path) as fp:
        soup = BeautifulSoup(fp, "html.parser")
    content = soup.get_text(separator=" ", strip=True).replace("\n", "")
    return re.sub("\s\s+", " ", content)


def create_preprocessed_data_csv(html_pages_folder_path: str = HTML_PAGES_FOLDER_PATH) -> None:
    """This function preprocesses the html pages dataset and saves the data to a csv file.
    The preprocessing consists of converting html to string content, removing special characters and duplicates.
    The function also checks if a page has already been preprocessed (exists in the csv) to avoid extra computations.

    :param html_pages_folder_path:
    :return:
    """
    if os.path.isfile(PREPROCESSED_DATA_CSV_PATH):
        preprocessed_df = pd.read_csv(PREPROCESSED_DATA_CSV_PATH)
        preprocessed_df["content"].fillna("", inplace=True)
    else:
        preprocessed_df = pd.DataFrame(columns=["filename", "content"])

    preprocessed_pages_list = []
    for html_file in glob(html_pages_folder_path + "*.html"):
        filename = html_file.split("/")[-1]
        if filename not in preprocessed_df["filename"].tolist():
            page_content = convert_html_to_text(html_file)
            page_content = "".join(
                ch for ch in page_content if ch not in ["-", "+", "@", "^", ":", "!", "?", "|", "(", ")"]
            )  # removing special characters
            preprocessed_pages_list.append({"filename": filename, "content": page_content})

    preprocessed_df = pd.concat([preprocessed_df, pd.DataFrame(preprocessed_pages_list)], ignore_index=True)
    preprocessed_df = preprocessed_df.drop_duplicates(subset=["content"])
    preprocessed_df.to_csv(PREPROCESSED_DATA_CSV_PATH, index=False)


if __name__ == "__main__":
    create_preprocessed_data_csv()
