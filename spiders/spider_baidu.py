# @Author: Ivan
# @LastEdit: 2020/9/23
import os
import json
import urllib
import requests  # install

width, height = '', ''
page = 10  # 需要下载的页数
rn = 50  # 每一页数量

keyword_list = ['鱼香茄子', '西红柿炒鸡蛋', '红烧肉', '椒盐虾', '土豆丝', '白菜', '青菜', '冬瓜排骨汤']
path = './raw_data/baidu'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36',
    'Referer': 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1551240778642_R\
		&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E8%A5%BF%E7%BA%A2%E6%9F%BF%E7%82%92%E9%B8%A1%E8%9B%8B'
}
"""
http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&query
	Word=%E8%A5%BF%E7%BA%A2%E6%9F%BF%E7%82%92%E9%B8%A1%E8%9B%8B&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=
	&st=&z=&ic=&hd=&latest=&copyright=&word=%E8%A5%BF%E7%BA%A2%E6%9F%BF%E7%82%92%E9%B8%A1%E8%9B%8B&s=
	&se=&tab=&width=&height=&face=&istype=&qc=&nc=1&fr=&expermode=&force=&pn=60&rn=60
"""


def download(lis, path, keyword):
    number = 1
    c_path = path + '\\' + keyword
    if not os.path.exists(path):
        os.makedirs(path)  # must makedirs here!
    if not os.path.exists(c_path):
        os.mkdir(c_path)
    for url in lis:
        r = requests.get(url, headers=headers, stream=True)
        with open(c_path + '\\' + keyword+str(number) + '.jpg', 'wb') as f:
            for chunk in r.iter_content(chunk_size=32):
                f.write(chunk)
        print(keyword, number, url, 'downloaded!')
        number = number + 1
    print('Downloading successfully!')


def download_all(keyword):
    number = 1  # 图片数量
    k = urllib.parse.quote(keyword)  # 关键字转码
    lis = []
    for i in range(page):
        pn = i * rn
        url = 'http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=' \
              + str(k) + '&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&word=' \
              + str(k) + '&s=&se=&tab=&width=' + str(width) + '&height=' + str(height) + \
              '&face=&istype=&qc=&nc=1&fr=&expermode=&force=&pn=' + \
              str(pn) + '&rn=' + str(rn)

        Referer = 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=\
		&sf=1&fmq=1551240778642_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype\
		=2&ie=utf-8&word=' + str(k)
        headers['Referer'] = Referer

        r = requests.get(url, headers=headers)
        try:
            j = json.loads(r.text)  # json字符串转换成字典
        except:
            print('json loads error!')
            continue
        data = j['data']
        for d in data:
            try:
                thumb_url = d['thumbURL']  # 找到图片url
                # print(thumbURL+' '+str(number))
                lis.append(thumb_url)  # 将图片url添加到lis列表
                print(thumb_url + ' ' + str(number))
                number = number + 1
            except:
                print('find thumbURL error!')
                continue
    download(lis, path, keyword)


def main():
    for keyword in keyword_list:
        download_all(keyword)


if __name__ == '__main__':
    main()
