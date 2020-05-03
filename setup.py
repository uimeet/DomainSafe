# coding=utf-8

import os, sys
import web
import settings
from core.libs.application import AmengApplication

sys.path.append(os.path.join(os.getcwd()))

from apps.api import app as api_app

# 配置网站的根路径
# 尝试解决web.seeother跳转不从根路径开始的问题
os.environ['SCRIPT_NAME'] = ''
os.environ['REAL_SCRIPT_NAME'] = ''


def env():
    import json

    return json.dumps({k: str(v) for k, v in web.ctx.env.items()})


urls = (
    '/env', env,

    # Ajax 调用命令
    '/async-cmd/([\w\-]+)', 'apps.common.views.AsyncCommand',
    # 验证码
    '/verimage', 'apps.common.views.VerImage',
    # API
    '/api', api_app,
)


def loadhook():
    pass


bm = AmengApplication(urls, locals())
bm.add_processor(web.loadhook(loadhook))
