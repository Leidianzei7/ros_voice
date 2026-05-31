#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pypinyin import lazy_pinyin
from .config import WAKE_WORD, WAKE_WORD_SHORT, _WW_LEN, _WWS_LEN, _WW_PINYIN, _WWS_PINYIN


def find_wake_word(text):
    """返回 (起始位置, 匹配长度)，找不到返回 (-1, 0)。优先匹配双"小智"，再匹配单"小智"。"""
    if WAKE_WORD in text:
        return text.index(WAKE_WORD), _WW_LEN
    for i in range(len(text) - _WW_LEN + 1):
        if "".join(lazy_pinyin(text[i : i + _WW_LEN])) == _WW_PINYIN:
            return i, _WW_LEN
    if WAKE_WORD_SHORT in text:
        return text.index(WAKE_WORD_SHORT), _WWS_LEN
    for i in range(len(text) - _WWS_LEN + 1):
        if "".join(lazy_pinyin(text[i : i + _WWS_LEN])) == _WWS_PINYIN:
            return i, _WWS_LEN
    return -1, 0
