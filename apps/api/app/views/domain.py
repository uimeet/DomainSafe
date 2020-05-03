
import os
import asyncio
from core import utils, messages

from apps import async_response
from apps.api import ApiViewBase

import settings

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
        return { 'domain': domain, 'polluted': result == 0 }
