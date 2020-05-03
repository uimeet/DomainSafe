from core.libs.application import AmengApplication

from apps.api.app import app as app_app

urls = (
    # ==== APP ====
    '/app', app_app,
)

app = AmengApplication(urls, locals())


class ApiViewBase(object):
    "API视图基类"

    def OPTIONS(self, *kargs, **kwargs):
        return ''
