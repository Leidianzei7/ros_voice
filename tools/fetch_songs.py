#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 archive.org 拉取公有领域 / 开放授权音频，按「歌名.mp3」存入曲库。

只检索白名单合集（见 _COLLECTIONS），这些合集内的录音要么已进入公有领域
（Great 78 Project 收录的多为 1920–50 年代唱片），要么带明确的开放授权。
每条结果都会打印其授权信息，便于自行复核。

用法：
    python3 tools/fetch_songs.py --query "chinese"           # 搜中文老唱片
    python3 tools/fetch_songs.py --query "folk" --limit 10
    python3 tools/fetch_songs.py --list-only                 # 只看不下
    python3 tools/fetch_songs.py --query "opera" --dest /path/to/songs

下完重启 brain_node 即可生效（指令集在模块导入时构建）。
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

_UA = "ros_voice-song-fetcher/1.0 (elderly-care robot demo)"

# 白名单合集：公有领域或开放授权
_COLLECTIONS = [
    "78rpm",              # Great 78 Project：历史唱片，多数已进入公有领域
    "audio_music",        # 用户上传音乐，配合 licenseurl 过滤
    "musopen",            # 公版古典乐
]

_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _search(query, collection, limit):
    q = f'collection:({collection}) AND mediatype:(audio)'
    if query:
        q += f' AND ({query})'
    params = urllib.parse.urlencode({
        "q": q, "rows": limit, "page": 1, "output": "json",
    }, safe=":()")
    params += "".join(f"&fl[]={f}" for f in
                      ("identifier", "title", "creator", "licenseurl", "year"))
    url = f"https://archive.org/advancedsearch.php?{params}"
    data = json.loads(_get(url))
    return data.get("response", {}).get("docs", [])


def _pick_audio(identifier):
    """列出条目里的 mp3，按体积从小到大（小的多为 VBR/64Kbps，省流量）。

    返回候选列表而非单个——archive.org 个别文件会 500，需要换下一个重试。
    """
    meta = json.loads(_get(f"https://archive.org/metadata/{identifier}"))
    cands = [f for f in meta.get("files", [])
             if f.get("name", "").lower().endswith(".mp3") and f.get("size")]
    cands.sort(key=lambda f: int(f.get("size", 1 << 62)))
    return [(f["name"], int(f.get("size", 0))) for f in cands]


def _safe_name(title, fallback):
    name = _BAD_CHARS.sub("", (title or "").strip()).strip(". ")
    name = re.sub(r"\s+", " ", name)[:80]
    return name or fallback


def main():
    ap = argparse.ArgumentParser(description="从 archive.org 拉公版歌曲到曲库")
    ap.add_argument("--query", default="", help="检索词，如 chinese / folk / opera")
    ap.add_argument("--collection", default=None,
                    help=f"指定合集，默认依次尝试 {_COLLECTIONS}")
    ap.add_argument("--limit", type=int, default=8, help="最多下载几首")
    ap.add_argument("--dest", default=None, help="目标目录，默认 <包目录>/songs")
    ap.add_argument("--list-only", action="store_true", help="只列出，不下载")
    args = ap.parse_args()

    dest = args.dest or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "songs")
    os.makedirs(dest, exist_ok=True)

    cols = [args.collection] if args.collection else _COLLECTIONS
    docs, seen = [], set()
    for c in cols:
        if len(docs) >= args.limit:
            break
        try:
            # 多取一些，因为下面要按授权过滤
            for d in _search(args.query, c, (args.limit - len(docs)) * 4):
                if d.get("identifier") in seen:
                    continue
                # 78rpm/musopen 整个合集即公版；其余合集必须有明确授权声明，
                # 「未标注」一律丢弃——宁可少下，不碰来路不明的。
                if c not in ("78rpm", "musopen") and not d.get("licenseurl"):
                    continue
                seen.add(d["identifier"])
                d["_collection"] = c
                docs.append(d)
                if len(docs) >= args.limit:
                    break
        except Exception as e:
            print(f"[搜索失败] {c}: {e}", file=sys.stderr)

    if not docs:
        print("没搜到结果，换个 --query 试试（英文关键词命中率更高）")
        return 1

    print(f"命中 {len(docs)} 条：\n")
    ok = 0
    for i, d in enumerate(docs, 1):
        ident = d["identifier"]
        title = d.get("title") or ident
        if isinstance(title, list):
            title = title[0]
        lic = d.get("licenseurl") or ("公有领域(78rpm 历史录音)"
                                      if d["_collection"] == "78rpm" else "未标注")
        print(f"[{i}] {title}")
        print(f"    创作者: {d.get('creator','?')}  年份: {d.get('year','?')}")
        print(f"    授权: {lic}")
        print(f"    来源: https://archive.org/details/{ident}")

        if args.list_only:
            continue

        try:
            cands = _pick_audio(ident)
            if not cands:
                print("    ⤷ 跳过：该条目没有 mp3\n")
                continue
            out = os.path.join(dest, _safe_name(title, ident) + ".mp3")
            if os.path.exists(out):
                print("    ⤷ 已存在，跳过\n")
                continue

            saved = False
            for fname, size in cands[:3]:          # 个别文件会 500，换下一个
                url = (f"https://archive.org/download/{ident}/"
                       f"{urllib.parse.quote(fname)}")
                print(f"    ⤷ 下载 {size/1024/1024:.1f}MB ...",
                      end="", flush=True)
                try:
                    data = _get(url, timeout=120)
                except Exception as e:
                    print(f" 失败({e})，换下一个")
                    continue
                with open(out, "wb") as fp:
                    fp.write(data)
                print(f" 存为 {os.path.basename(out)}\n")
                saved = True
                ok += 1
                break
            if not saved:
                print("    ⤷ 该条目所有候选文件均下载失败\n", file=sys.stderr)
        except Exception as e:
            print(f"\n    ⤷ 处理失败: {e}\n", file=sys.stderr)

    if not args.list_only:
        print(f"完成：新增 {ok} 首 -> {dest}")
        print("提示：文件名即歌名，可自行改成中文；改完重启 brain_node 生效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
