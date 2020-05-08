
import web

from core.libs.application import AmengApplication

urls = (
    # 检测dns污染
    '/domain/pollute_check', 'apps.api.app.views.domain.DNSPolluteCheck',
    # 获取域名ip及title信息
    '/domain/ping_get', 'apps.api.app.views.domain.PingGet',
)

app = AmengApplication(urls, locals())