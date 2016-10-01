# -*- encoding:utf-8 -*-
import threading
import PIL.Image
import ZLog
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
import ZCommonUtil
from Decorator import warnings_filter

__author__ = 'BBFamily'


class SpiderProxy(object):
    @classmethod
    def read_csv(cls):
        fn = '../gen/proxy/proxy_df'
        return pd.read_csv(fn, index_col=0)

    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_0) AppleWebKit/537.36 (KHTML, "
                                      "like Gecko) Chrome/32.0.1664.3 Safari/537.36"}
        self.proxy_list = list()
        self.session = requests.Session()

    def spider_proxy360(self):
        r = self.session.get('http://www.proxy360.cn/default.aspx', headers=self.headers)
        soup = BeautifulSoup(r.text, "lxml")
        ip_list = soup.select('#ctl00_ContentPlaceHolder1_upProjectList > div > div > span[class="tbBottomLine"]')

        for ind in np.arange(0, len(ip_list), 2):
            ip = ip_list[ind].string.strip()
            port = ip_list[ind + 1].string.strip()
            self.proxy_list.append({'ip': ip, 'port': port, 'proxy': '{}:{}'.format(ip, port), 'type': 'HTTP'})

    def spider_xicidaili(self):
        r = self.session.get('http://www.xicidaili.com/', headers=self.headers)
        soup = BeautifulSoup(r.text, "lxml")
        ip_pattern = re.compile('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', re.S)
        type_pattern = re.compile('HTTP|socks4/5', re.S)
        port_pattern = re.compile('^\d{1,5}$', re.S)
        for tr in soup.find_all("tr"):
            if tr.has_attr('class') and (tr['class'][0] == 'odd' or tr['class'][0] == ''):
                ip = None
                port = None
                proxy_type = None
                for td in tr.find_all('td'):
                    if td.string and re.match(ip_pattern, td.string):
                        ip = td.string
                    if td.string and re.match(type_pattern, td.string):
                        proxy_type = td.string
                    if td.string and re.match(port_pattern, td.string):
                        port = td.string
                if ip is not None and port is not None and proxy_type is not None:
                    self.proxy_list.append(
                        {'ip': ip, 'port': port, 'proxy': '{}:{}'.format(ip, port), 'type': proxy_type})

    @warnings_filter
    def do_thread_work(self, proxy, checked_list, thread_lock):
        if proxy['type'] == 'HTTP':
            proxy_dict = dict(http='http://{}'.format(proxy['proxy']),
                              https='http://{}'.format(proxy['proxy']))
        else:
            proxy_dict = dict(http='socks5://{}'.format(proxy['proxy']),
                              https='socks5://{}'.format(proxy['proxy']))

        try:
            # r = requests.post("https://www.baidu.com/", headers=self.headers, proxies=proxy_dict, timeout=15,
            #                   verify=False)
            img_url = 'http://picm.bbzhi.com/dongwubizhi/labuladuoxunhuiquanbizhi/animal_' \
                      'labrador_retriever_1600x1200_44243_m.jpg'

            enable_stream = False
            if enable_stream:
                response = requests.get(img_url, headers=self.headers, proxies=proxy_dict, timeout=15, stream=True)
                if response.status_code == 200:
                    test_name = '../gen/check_proxy.jpg'
                    with open(test_name, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)
                                f.flush()

                        check_img = PIL.Image.open(test_name)
                        check_img.close()
            else:
                response = requests.get(img_url, headers=self.headers, proxies=proxy_dict, timeout=(10, 20))
                if response.status_code == 200:
                    test_name = '../gen/check_proxy.jpg'
                    with open(test_name, 'wb') as f:
                        f.write(response.content)
                        f.flush()
                    check_img = PIL.Image.open(test_name)
                    check_img.close()
        except Exception as e:
            # ZLog.exception(e)
            return
        with thread_lock:
            ZLog.info('{} check ok'.format(proxy['proxy']))
            checked_list.append(proxy)

    def check_proxy(self):
        checked_list = list()
        thread_lock = threading.RLock()
        thread_array = []
        for proxy in self.proxy_list:
            # self.do_thread_work(proxy, checked_list, thread_lock)
            t = threading.Thread(target=self.do_thread_work, args=(
                proxy,
                checked_list,
                thread_lock,))
            t.setDaemon(True)
            t.start()
            thread_array.append(t)

        for t in thread_array:
            t.join()

        self.proxy_list = checked_list
        ZLog.info('proxy_list len={}'.format(len(self.proxy_list)))

    def save_csv(self):
        self.proxy_df = pd.DataFrame(self.proxy_list)
        fn = '../gen/proxy/proxy_df'
        ZCommonUtil.ensure_dir(fn)
        self.proxy_df.to_csv(fn, columns=self.proxy_df.columns, index=True)
