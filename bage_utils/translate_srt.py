import json
from urllib.parse import quote

import requests
import srt
from tqdm import tqdm

# from google.cloud import translate
APIKey = "AIzaSyAPdRo0-dFQoKPp5GbB6FyVPIspt-ntjKU"  # Here is your API key
Url = "https://translation.googleapis.com/language/translate/v2?key="


def translate(queryString, source, target, mode="baidu"):
    if mode == "baidu":
        # print('queryString:', queryString)
        translation = translate_baidu(queryString, source, target)
    elif mode == "google":
        translation = translate_google(queryString, source, target)
    else:
        translation = ''
    return translation


def translate_google(queryString, source, target):
    requestURL = Url + APIKey + "&source=" + source + "&target=" + target + "&q=" + queryString
    response = requests.get(requestURL)
    # print('requestURL:', requestURL)
    translation = json.loads(response.text)["data"]["translations"][0]['translatedText']
    return translation


def translate_baidu(query, source, target):
    url = "http://crashcourse.club/api/translate/q=%s&from=%s&to=%s" % (quote(query), source, target)
    response = requests.get(url)
    try:
        translated = response.json()["trans_result"][0]["dst"]
    except:
        translated = ""
    return translated


def process(mode, input_file, output_file, source, target):
    with open(input_file, 'rt', encoding='utf-8', errors='ignore') as f:
        data = f.read()
        f.close()

    subs = list(srt.parse(data))

    for i, k in enumerate(tqdm(subs)):
        text = k.content
        translation = translate(text, source, target, mode)
        k.content = translation

        if i % 100 == 0:
            print(f'{i/len(subs)*100:.0f}% "{text}" -> "{translation}"')

    srtTranslated = srt.compose(subs)
    # write the srt file translated...
    with open(output_file, 'xt', encoding='utf-8', errors='ignore') as f:
        f.write(data)
        f.write(srtTranslated)
        f.close()


if __name__ == '__main__':
    # print(translate('living in Columbus, Ohio, with Aunt Alice.', 'en', 'ko', mode='google'))
    # """
    #     translate a subtitle file from a language to another.
    #     Most of the code to use google translate API was taken there :
    #       https://github.com/zhendi/google-translate-cli
    #
    #     usage:
    #       translate-srt.py input_file output_file input_language output_language
    #
    #     example:
    #       translate-srt.py ./titanic-en.srt ./titanic-fr.srt en fr
    # """
    process('google', '/Users/bage/Downloads/ReadyPlayerOne.en.srt', '/Users/bage/Downloads/ReadyPlayerOne.ko.srt', 'en', 'ko')
