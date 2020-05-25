from factories import factories
from pathlib import Path
import json
import spacy
import os
import sys
import toml
import shutil
from zipfile import ZipFile
import requests
from urllib3.exceptions import InsecureRequestWarning

# Suppress only the single warning from urllib3 needed.
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

DEFAULT_INDEX_URL = "https://raw.githubusercontent.com/marc-chan/textcat_trainer/master/index.toml"
BASE_MODELS = ['glove', 'word2vec', 'distilbert']

MODEL_ALIASES = {
    'basic': 'en_core_web_sm-2.2.5',
    'glove': 'en_core_web_lg-2.2.5',
    'word2vec': 'word2vec',
    'distilbert': 'en_trf_distilbertbaseuncased_lg-2.2.0',
}

PATHS = []
homedir = Path.home()
basedir = Path(sys.prefix)

if "APPENGINE_RUNTIME" not in os.environ:
    PATHS.append(homedir / str("tetris_data"))

if sys.platform.startswith("win"):
    # Common locations on Windows:
    PATHS += [
        basedir / str("tetris_data"),
        basedir / str("share") / str("tetris_data"),
        basedir / str("lib") / str("tetris_data"),
        Path(os.environ.get(str("APPDATA"), str("C:\\"))) / str("tetris_data"),
        Path(str("C:/tetris_data")),
        Path(str("D:/tetris_data")),
        Path(str("E:/tetris_data")),
    ]
else:
    # Common locations on UNIX & OS X:
    PATHS += [
        basedir / str("tetris_data"),
        basedir / str("share") / str("tetris_data"),
        basedir / str("lib") / str("tetris_data"),
        Path(str("/usr/share/tetris_data")),
        Path(str("/usr/local/share/tetris_data")),
        Path(str("/usr/lib/tetris_data")),
        Path(str("/usr/local/lib/tetris_data")),
    ]

def find(path_or_name):
    #First check against MODEL_ALIASES
    is_included = path_or_name in MODEL_ALIASES
    new_path_or_name = MODEL_ALIASES.get(path_or_name, path_or_name)
    path = Path(new_path_or_name)
    if path.exists():
        return path
    else:
        for dir_ in PATHS:
            dirpath = Path(dir_) / new_path_or_name
            if dirpath.exists():
                return dirpath
    if is_included:
        raise FileNotFoundError(f"Model {path_or_name} has not been downloaded. Try tetris.download('{path_or_name}')")
    raise ValueError("No model found.")
    
def load(path_or_name, typ=None):
    path = find(path_or_name)
    if (path / "meta.json").exists():
        meta = json.load((path/"meta.json").open('r'))
        mtype = meta.get('type')
        if typ and typ != mtype:
            raise ValueError("No model found.")
        if mtype:
            loader = factories.get(mtype)
            return loader(path)
        return spacy.load(path)
    else:
        raise ValueError("No model found.")

def is_dir_writable(dir_):
    # Ensure that it exists.
    if not dir_.exists():
        return False
    return os.access(dir_, os.W_OK | os.X_OK)

def default_download_dir():
    if "APPENGINE_RUNTIME" in os.environ:
            return

    # Check if we have sufficient permissions to install in a
    # variety of system-wide locations.
    for dir_ in PATHS:
        if dir_.exists() and is_dir_writable(dir_):
            return dir_

    # On Windows, use %APPDATA%
    if sys.platform == "win32" and "APPDATA" in os.environ:
        homedir = Path(os.environ["APPDATA"])

    # Otherwise, install in the user's home directory.
    else:
        homedir = Path.home()
        if str(homedir) == "~/":
            raise ValueError("Could not find a default download directory")

    # append "tetris_data" to the home directory
    return homedir / "tetris_data"

def get_resource_info(resource_name, index=None, index_url=None):
    if index:
        for k in index.keys():
            if isinstance(index[k], dict):
                for k_ in index[k].keys():
                    if k_ == resource_name:
                        return index[k][k_]['filename'], index[k][k_]['resource_url']
    elif index_url:
        index = toml.loads(requests.get(index_url, verify=False).text)
        return get_resource_info(resource_name, index=index)
    else:
        raise ValueError("Either `index` or `index_url` must be provided")


def download(resource_name, path=None, index=None, index_url=None):
    if path:
        path = Path(path)
    else:
        path = default_download_dir()

    if not path.exists():
        path.mkdir()

    if not index and not index_url:
        index_url = DEFAULT_INDEX_URL
    resource_filename, resource_url = get_resource_info(resource_name, index=index, index_url=index_url)
    resource_filepath = path / resource_filename
    print(f"Downloading {resource_filename} to {resource_filepath}")
    with requests.get(resource_url, stream=True, verify=False) as r:
        with open(resource_filepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    print(f"Unzipping...")
    if resource_filename.endswith(".zip"):
        with ZipFile(resource_filepath, 'r') as zip_ref:
            zip_ref.extractall(path)

    return

