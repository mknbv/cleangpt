"""Loads a random article from SEP and preprocesses it for training."""
import os
import requests
from bs4 import BeautifulSoup


URL = "https://plato.stanford.edu/cgi-bin/encyclopedia/random"
DATA_PATH = "../data/"


def get_random_page(data_path=DATA_PATH, timeout=1):
  """Returns the contents of a random url page."""
  response = requests.get(URL, timeout=timeout)
  response.raise_for_status()
  soup = BeautifulSoup(response.text, "html.parser")
  content = str(soup.find("div", id="aueditable"))
  if data_path is not None:
    filepath = os.path.join(data_path, response.url.split("/")[-2] + ".html")
    with open(filepath, "w", encoding="utf8") as outfile:
      outfile.write(content)
  return content
