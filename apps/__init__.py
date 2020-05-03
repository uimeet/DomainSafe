import web

from core import utils

RESPONSE_FIELDS = ('code', 'message', 'data')


def async_response(func):
    """
    异步响应
    """

    def wrapper(self, *args, **kw):
        # 请求开始时间
        start_ts = utils.timestamp()

        values = func(self, *args, **kw)

        web.header('Content-type', 'application/json')

        if not isinstance(values, (tuple, list)):
            values = [values]

        # data = { RESPONSE_FIELDS[i]: value for i, value in enumerate(values) if i < 3 }
        data = {field: values[i] if i < len(values) else None for i, field in enumerate(RESPONSE_FIELDS)}
        # 包含服务器时间
        data['time'] = int(utils.timestamp()) - 2
        data['ms'] = int((utils.timestamp() - start_ts) * 1000)
        data = utils.json_dumps(data, utils.JsonEncoder)
        # 是否提供第四个参数
        # 如果提供了第四个参数, 则是一个 jsonp callback 调用
        if len(values) == 4:
            return '%s(%s)' % (values[3], data)

        return data

    return wrapper
