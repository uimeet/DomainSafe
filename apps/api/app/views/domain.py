import os
import asyncio
import tldextract
from requests_html import HTMLSession
import traceback
from core import utils, messages

from apps import async_response
from apps.api import ApiViewBase

import settings


class PingGet(ApiViewBase):
    "获取给定域名的IP和Title"

    def __init__(self):
        # 是否使用代理
        self.proxy = False

    @async_response
    def GET(self):
        inp = utils.Input()

        domains = inp.split('domains')
        if not domains:
            return messages.ArgumentInvalid

        self.proxy = inp.boolean('proxy')

        result = asyncio.run(asyncio.wait([self.ping_get(domain) for domain in domains]))
        return 0, 'success', [r.result() for r in result[0]]

    async def ping_get(self, domain):
        if not domain.startswith('http'):
            domain = f'http://{domain}'

        if domain.endswith('/'):
            domain = domain[:-1]
        # 获取ip地址
        p = self.get_ip(domain)
        # 获取标题
        title = self.get_title(domain)

        d = tldextract.extract(domain)

        return { 'domain': f'{d.domain}.{d.suffix}', 'ip': p, 'title': title, 'url': domain }

    def get_ip(self, domain):
        """
        获取IP地址
        :param domain:
        :return:
        """
        domain = domain.replace('http://', '').replace('https://', '')
        lines = os.popen(f'nslookup {domain}').readlines();

        ips = None
        for line in lines:
            if line.startswith('Name:') and ips is None:
                ips = []
            if line.startswith('Address:') and ips is not None:
                ips.append(line.split(':')[-1].strip())

        return ips or []

    def get_title(self, domain):
        """
        获取域名标题
        :param domain:
        :return:
        """
        proxies = None
        if self.proxy:
            proxies = {
                'http': 'socks5://127.0.0.1:1089',
                'https': 'socks5://127.0.0.1:1089',
            }

        session = HTMLSession()
        try:
            r = session.get(domain, proxies=proxies)

            title = r.html.find('title', first=True)
            return title.text if title else '<未获取到title>'
        except:
            traceback.print_exc()
            return '<Title获取失败>'


class DNSPolluteCheck(ApiViewBase):
    "DNS 污染检测"

    @async_response
    def GET(self):
        inp = utils.Input()

        domains = inp.split('domains')
        if not domains:
            return messages.ArgumentInvalid

        result = asyncio.run(asyncio.wait([self.check(domain) for domain in domains]))
        return 0, 'success', [r.result() for r in result[0]]

    async def check(self, domain):
        """
        检查给定域名
        :param domain:
        :return:
        """
        result = utils.intval(os.popen(f'{settings.APP_ROOT}/bin/dnscheck {domain}').readline().strip())
        return {'domain': domain, 'polluted': result == 0}
