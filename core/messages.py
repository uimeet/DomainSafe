# coding=utf-8

from web.utils import storage


Success = (0, u'success',)
ArgumentInvalid = (101, u'参数无效',)
MissArguments = (102, u'缺少参数',)
NotFound = (404, u'无数据')
TooManyRequests = (406, '请求太快了')
NoOperation = (1000, u'未执行任何操作')

NoLogin = (401, u'登录超时')
NoPermission = (403, u'无权限访问')
MethodNotAllowed = (405, 'Method Not Allowed')

UnknownTagError = (1001, u'未知标签')
UnknownElementError = (1002, u'未知元素')

StopRepeatSubmit = (1003, '提交太快了，歇一会儿吧')

UnknownError = (99999, u'未知错误')

