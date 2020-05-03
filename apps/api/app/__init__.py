
import web

from core.libs.application import AmengApplication

urls = (
    # 检测dns污染
    '/domain/pollute_check', 'apps.api.app.views.domain.DNSPolluteCheck',
)

app = AmengApplication(urls, locals())