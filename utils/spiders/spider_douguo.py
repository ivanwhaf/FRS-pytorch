# @Author: Ivan
# @LastEdit: 2020/9/4
import os
from urllib import parse
import requests  # install
from bs4 import BeautifulSoup as bs  # install

# https://www.douguo.com/search/recipe/%E7%B3%96%E9%86%8B%E6%8E%92%E9%AA%A8/0/0
douguo_api = 'https://www.douguo.com/search/recipe/'
keyword_list = ['鱼香茄子', '西红柿炒鸡蛋', '红烧肉', '椒盐虾', '土豆丝', '白菜', '青菜', '冬瓜排骨汤']
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/72.0.3626.109 Safari/537.36'}
path = './raw_data/douguo'
page = 25  # 要下载的页数，每页最多20张图片


def download(lis, path, keyword):
    number = 1
    c_path = path + '\\' + keyword
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(c_path):
        os.mkdir(c_path)
    for url in lis:
        r = requests.get(url, headers=headers, stream=True)
        with open(c_path + '\\' + keyword+str(number) + '.jpg', 'wb') as f:
            for chunk in r.iter_content(chunk_size=32):
                f.write(chunk)
        print(number, url, 'downloaded!')
        number = number + 1
    print('Downloading successfully!')


def download_all(keyword):
    k = parse.quote(keyword)
    lis = []
    for p in range(page):
        url = douguo_api + k + '/0/' + str(20 * p)
        r = requests.get(url, headers=headers)
        soup = bs(r.text, 'html.parser')
        cook_list = soup.find('ul', class_='cook-list')
        for li in cook_list.find_all('li'):
            img = li.find('a', class_='cook-img').find('img')
            img_url = img.get('src')
            # print(img_url)
            lis.append(img_url)
    download(lis, path, keyword)


def main():
    for keyword in keyword_list:
        download_all(keyword)


if __name__ == '__main__':
    main()
