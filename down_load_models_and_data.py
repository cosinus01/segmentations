import requests
import urllib
import zipfile
import os
import json
import sys

try:
    with open("urls.json", "r") as f:
        URLS = json.load(f)
except FileNotFoundError:
    print("urls.json отсутствует!")
    sys.exit(0)



http_proxy  = ""
https_proxy = ""


proxyDict = { 
              "http"  : http_proxy, 
              "https" : https_proxy, 
            }

API_ENDPOINT = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}'

def _get_real_direct_link(sharing_link):
    pk_request = requests.get(API_ENDPOINT.format(sharing_link), proxies=proxyDict, verify=False)
    return pk_request.json().get('href')

def _extract_filename(direct_link):
    for chunk in direct_link.strip().split('&'):
        if chunk.startswith('filename='):
            return chunk.split('=')[1]
    return None

def download_yadisk_link(sharing_link, path="images/", unzip=False):
    direct_link = _get_real_direct_link(sharing_link)
    if direct_link:
        filename = path + _extract_filename(direct_link)
        download = requests.get(direct_link, proxies=proxyDict, verify=False)
        with open(filename, 'wb') as out_file:
            out_file.write(download.content)
        if unzip:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(path)
            os.remove(filename) 
            print('Downloaded "{}" to "{}"'.format(sharing_link, filename[:-4]))
        else:
            print('Downloaded "{}" to "{}"'.format(sharing_link, filename))
    else:
        print('Failed to download "{}"'.format(sharing_link))

download_yadisk_link(URLS["water_data_set"], "images/", unzip=True)
download_yadisk_link(URLS["urban_data_set"], "images/", unzip=True)
download_yadisk_link(URLS["forest_data_set"], "images/", unzip=True)
download_yadisk_link(URLS["models"], "", unzip=True)