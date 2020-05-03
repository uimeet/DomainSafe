# coding=utf-8

import logging
import logging.config

import traceback

# 初始化日志对象
try:
    logging.config.fileConfig('%s/logger.conf' % '/'.join(__file__.split('/')[:-1]))
except KeyError:
    pass


def debug(text, *args, **kwargs):
    return logging.debug(text, *args, **kwargs)


def log(text, *args, **kwargs):
    return logging.info(text, *args, **kwargs)


def error(text, *args, **kwargs):
    return logging.error(text, *args, **kwargs)


def warning(text, *args, **kwargs):
    return logging.warning(text, *args, **kwargs)


def print_exception():
    "打印异常"
    error(traceback.format_exc())


def format_log(tag, **kwargs):
    """
    格式化日志输出内容
    :param tag:
    :param values:
    :return:
    """
    texts = [
        f'#################### {tag} ####################',
    ]

    if kwargs:
        for key, value in kwargs.items():
            texts.append(f'# {key}: {value}')

    return '\r\n'.join(texts)
