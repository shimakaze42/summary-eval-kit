#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文書要約エージェント評価キット v1（決定的指標のみ・LLM呼び出しなし）

章別要約（summary.md）を、対応する原文（book.txt の章ページ範囲）と機械的に照合し、
  1. 数字チェック      : 要約中の数値表現が原文に存在するか（なければハルシネーション候補）
  2. 固有名詞チェック  : 要約中のカタカナ語・短い鉤括弧用語が原文に存在するか
  3. 章カバー率        : 要約が付いている章の割合、章ごとの要約長／原文長
を章ごとに出力する。

設計方針:
  - 「候補」と「確定」を分ける。表記ゆれ・OCR誤り・パラフレーズで原文に
    文字列として見つからないだけのケースがあるため、未一致はすべて
    「ハルシネーション候補」として人が見る前提で列挙する。
  - 照合は表記を正規化してから行う（NFKC、空白除去、カンマ除去）。
  - 原文・要約データは読み取りのみ。改変しない。

使い方:
  python3 eval_summary.py [--data-dir DIR] [--out results_v1.md] [--json results_v1.json]
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import OrderedDict
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / "workspace" / "kindle_summarizer" / "jinzai"

# ---------------------------------------------------------------------------
# 読み込みと分割
# ---------------------------------------------------------------------------

def load_pages(book_path: Path) -> "OrderedDict[int, str]":
    """book.txt を [p.N] マーカーでページ単位に分割する。"""
    text = book_path.read_text(encoding="utf-8")
    parts = re.split(r"^\[p\.(\d+)\]\s*$", text, flags=re.MULTILINE)
    pages: "OrderedDict[int, str]" = OrderedDict()
    # parts = [先頭の残り, page_no, body, page_no, body, ...]
    for i in range(1, len(parts) - 1, 2):
        pages[int(parts[i])] = parts[i + 1]
    return pages


def load_summary_sections(summary_path: Path) -> "OrderedDict[str, str]":
    """summary.md を '## ' 見出しで節に分割する。"""
    text = summary_path.read_text(encoding="utf-8")
    parts = re.split(r"^##\s+(.+)$", text, flags=re.MULTILINE)
    sections: "OrderedDict[str, str]" = OrderedDict()
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i].strip()] = parts[i + 1]
    return sections


def match_section_to_chapter(heading: str, chapters: list) -> dict | None:
    """要約の見出しを chapters.json の章に対応づける。"""
    m = re.match(r"Chapter\s+(\d+)\b", heading)
    if m:
        n = m.group(1)
        for ch in chapters:
            if re.match(rf"Chapter\s+{n}\b", ch["title"]):
                return ch
        return None
    for ch in chapters:
        if ch["title"] == heading:
            return ch
    return None


# ---------------------------------------------------------------------------
# 正規化とクリーニング
# ---------------------------------------------------------------------------

def normalize_for_match(text: str) -> str:
    """照合用に表記を寄せる: NFKC、全空白除去、カンマ除去。"""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"\s+", "", t)
    t = t.replace(",", "").replace("、", "、")  # 数字内カンマ対策（読点は残す）
    return t


def clean_summary_text(text: str) -> str:
    """要約文から、内容でない要素（引用ページ注記・図ファイル名・Markdown記法）を除く。"""
    t = text
    t = re.sub(r"\[p[^\]]*\]", " ", t)                     # [p.16] [p.20, p.34] 等の引用注記
    t = re.sub(r"fig_p\d+_\d+\.png", " ", t)               # 図ファイル名
    t = re.sub(r"図表\s*\d+", "図表", t)                    # 図表005 等の図番号は内容の数字ではない
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)            # 画像
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)         # リンク
    t = re.sub(r"^#+\s*", "", t, flags=re.MULTILINE)
    return t


def clean_source_text(text: str) -> str:
    """原文から図表キャプション行を除く（fig_p0001 等の数字が誤一致するのを防ぐ）。"""
    t = re.sub(r"\[図表:[^\]]*\]", " ", text)
    t = re.sub(r"fig_p\d+_\d+\.png", " ", t)
    return t


# ---------------------------------------------------------------------------
# 抽出
# ---------------------------------------------------------------------------

UNITS = [
    "カ国", "万人", "億円", "万円", "千人",
    "時間", "項目", "ページ",
    "%", "年", "月", "日", "人", "社", "名", "件", "倍", "割", "円", "歳",
    "分", "秒", "つ", "個", "冊", "頁", "本",
]
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
NUMBER_RE = re.compile(rf"\d[\d,]*(?:\.\d+)?\s*(?:{_UNIT_ALT})?")
KATAKANA_RE = re.compile(r"[ァ-ヶー・]{2,}")
QUOTED_RE = re.compile(r"「([^「」]{2,12})」|『([^『』]{2,20})』")


def extract_numbers(text: str) -> list:
    """数値表現（数字＋直後の単位）を出現順に返す。重複は残す（総数＝出現数）。"""
    t = unicodedata.normalize("NFKC", text)
    out = []
    for m in NUMBER_RE.finditer(t):
        tok = re.sub(r"\s+", "", m.group(0)).replace(",", "")
        # 数字だけの断片（ページ番号の残骸など）を避けるため、最低1桁の数字を含むこと
        if re.search(r"\d", tok):
            out.append(tok)
    return out


def extract_terms(text: str) -> list:
    """固有名詞候補（カタカナ語と短い鉤括弧用語）を出現順・重複なしで返す。"""
    t = unicodedata.normalize("NFKC", text)
    terms: "OrderedDict[str, None]" = OrderedDict()
    for m in KATAKANA_RE.finditer(t):
        w = m.group(0).strip("・")
        if len(w) >= 2:
            terms.setdefault(w)
    for m in QUOTED_RE.finditer(t):
        w = (m.group(1) or m.group(2) or "").strip()
        if w:
            terms.setdefault(w)
    return list(terms.keys())


def find_context(text: str, needle: str, width: int = 40) -> str:
    """要約文中で needle が現れる周辺を短く切り出す（詳細表示用）。"""
    t = unicodedata.normalize("NFKC", text)
    i = t.find(unicodedata.normalize("NFKC", needle))
    if i < 0:
        return ""
    s = max(0, i - width)
    e = min(len(t), i + len(needle) + width)
    snippet = t[s:e].replace("\n", " ")
    return ("…" if s > 0 else "") + snippet + ("…" if e < len(t) else "")


# ---------------------------------------------------------------------------
# 評価本体
# ---------------------------------------------------------------------------

def evaluate(data_dir: Path) -> dict:
    chapters = json.loads((data_dir / "chapters.json").read_text(encoding="utf-8"))["chapters"]
    pages = load_pages(data_dir / "book.txt")
    sections = load_summary_sections(data_dir / "summary.md")

    def chapter_source(ch: dict) -> str:
        return "".join(pages.get(p, "") for p in range(ch["start"], ch["end"] + 1))

    # 要約節 → 章 の対応
    section_for_chapter: dict[str, str] = {}
    whole_book_sections: dict[str, str] = {}
    for heading, body in sections.items():
        ch = match_section_to_chapter(heading, chapters)
        if ch is not None:
            section_for_chapter[ch["title"]] = body
        else:
            whole_book_sections[heading] = body  # 「全体の要点」など

    book_all_norm = normalize_for_match(clean_source_text("".join(pages.values())))

    results = {"chapters": [], "whole_book_sections": [], "meta": {}}

    for ch in chapters:
        src_raw = chapter_source(ch)
        src_norm = normalize_for_match(clean_source_text(src_raw))
        body = section_for_chapter.get(ch["title"])
        entry = {
            "title": ch["title"],
            "pages": f"p.{ch['start']}-{ch['end']}",
            "source_chars": len(re.sub(r"\s+", "", src_raw)),
            "has_summary": body is not None,
        }
        if body is not None:
            cleaned = clean_summary_text(body)
            nums = extract_numbers(cleaned)
            terms = extract_terms(cleaned)
            num_detail = [
                {"token": n, "found": normalize_for_match(n) in src_norm,
                 "context": find_context(cleaned, n)}
                for n in nums
            ]
            term_detail = [
                {"token": w, "found": normalize_for_match(w) in src_norm}
                for w in terms
            ]
            entry.update({
                "summary_chars": len(re.sub(r"\s+", "", cleaned)),
                "length_ratio": (len(re.sub(r"\s+", "", cleaned)) /
                                 max(1, len(re.sub(r"\s+", "", src_raw)))),
                "numbers": num_detail,
                "terms": term_detail,
            })
        results["chapters"].append(entry)

    for heading, body in whole_book_sections.items():
        cleaned = clean_summary_text(body)
        nums = extract_numbers(cleaned)
        terms = extract_terms(cleaned)
        results["whole_book_sections"].append({
            "heading": heading,
            "summary_chars": len(re.sub(r"\s+", "", cleaned)),
            "numbers": [{"token": n, "found": normalize_for_match(n) in book_all_norm,
                         "context": find_context(cleaned, n)} for n in nums],
            "terms": [{"token": w, "found": normalize_for_match(w) in book_all_norm}
                      for w in terms],
        })

    covered = sum(1 for c in results["chapters"] if c["has_summary"])
    results["meta"] = {
        "total_chapters": len(chapters),
        "covered_chapters": covered,
        "total_pages": len(pages),
    }
    return results


# ---------------------------------------------------------------------------
# レポート生成
# ---------------------------------------------------------------------------

def pct(a: int, b: int) -> str:
    return f"{(100.0 * a / b):.1f}%" if b else "—"


def render_report(res: dict) -> str:
    m = res["meta"]
    lines = []
    lines.append("# 評価キット v1 結果 — 『図解 人材マネジメント入門』章別要約")
    lines.append("")
    lines.append("測っているもの: 章別要約に出てくる「数字」と「固有名詞（カタカナ語・短い鉤括弧用語）」が、"
                 "対応する章の原文に文字列として存在するか。存在しないものは**ハルシネーション候補**として数える"
                 "（表記ゆれ・OCR誤り・言い換えでも未一致になり得るため、候補≠確定）。LLM呼び出しなし、決定的指標のみ。")
    lines.append("")
    lines.append("## 全体サマリ")
    lines.append("")
    lines.append(f"- 章カバー率: 全{m['total_chapters']}章中 {m['covered_chapters']}章に要約あり "
                 f"({pct(m['covered_chapters'], m['total_chapters'])})。"
                 f"要約がない章: " +
                 "、".join(c["title"] for c in res["chapters"] if not c["has_summary"]))
    all_nums = [n for c in res["chapters"] if c["has_summary"] for n in c["numbers"]]
    all_terms = [t for c in res["chapters"] if c["has_summary"] for t in c["terms"]]
    nf = sum(1 for n in all_nums if n["found"])
    tf = sum(1 for t in all_terms if t["found"])
    lines.append(f"- 数字（章別要約全体）: {len(all_nums)}件中 {nf}件が原文に存在（精度 {pct(nf, len(all_nums))}）、"
                 f"候補 {len(all_nums) - nf}件")
    lines.append(f"- 固有名詞（章別要約全体）: {len(all_terms)}件中 {tf}件が原文に存在（精度 {pct(tf, len(all_terms))}）、"
                 f"候補 {len(all_terms) - tf}件")
    precisions = []
    for c in res["chapters"]:
        if c["has_summary"] and c["numbers"]:
            precisions.append(sum(1 for n in c["numbers"] if n["found"]) / len(c["numbers"]))
    if precisions:
        lines.append(f"- 章ごとの数字精度の分布: 最低 {min(precisions)*100:.0f}% 〜 最高 {max(precisions)*100:.0f}%"
                     f"（中央値 {sorted(precisions)[len(precisions)//2]*100:.0f}%）")
    lines.append("")
    lines.append("## 章ごとの結果")
    lines.append("")
    lines.append("| 章 | 原文字数 | 要約字数 | 長さ比 | 数字 一致/総数 | 数字精度 | 固有名詞 一致/総数 | 固有名詞精度 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in res["chapters"]:
        if not c["has_summary"]:
            lines.append(f"| {c['title']}（{c['pages']}） | {c['source_chars']:,} | — | — | — | — | — | — |")
            continue
        nn, nf2 = len(c["numbers"]), sum(1 for n in c["numbers"] if n["found"])
        tn, tf2 = len(c["terms"]), sum(1 for t in c["terms"] if t["found"])
        lines.append(f"| {c['title']}（{c['pages']}） | {c['source_chars']:,} | {c['summary_chars']:,} | "
                     f"{c['length_ratio']*100:.1f}% | {nf2}/{nn} | {pct(nf2, nn)} | {tf2}/{tn} | {pct(tf2, tn)} |")
    lines.append("")
    lines.append("「全体の要点」節は本全体を照合範囲として別枠で測った（章別表には含めない）:")
    for s in res["whole_book_sections"]:
        nn, nf2 = len(s["numbers"]), sum(1 for n in s["numbers"] if n["found"])
        tn, tf2 = len(s["terms"]), sum(1 for t in s["terms"] if t["found"])
        lines.append(f"- {s['heading']}: 数字 {nf2}/{nn}（{pct(nf2, nn)}）、固有名詞 {tf2}/{tn}（{pct(tf2, tn)}）")
    lines.append("")

    # ---- Chapter 1 詳細 ----
    ch1 = next((c for c in res["chapters"] if c["title"].startswith("Chapter 1 ")), None)
    if ch1 and ch1["has_summary"]:
        lines.append(f"## 詳細: {ch1['title']}（{ch1['pages']}）")
        lines.append("")
        lines.append("本人が目を通す前提の詳細版。この章の数字・固有名詞をすべて列挙し、原文にあったかを示す。")
        lines.append("")
        lines.append("### 数字")
        lines.append("")
        lines.append("| 数字 | 原文に存在 | 要約中の出現箇所 |")
        lines.append("|---|---|---|")
        for n in ch1["numbers"]:
            ctx = n["context"].replace("|", "｜")
            lines.append(f"| {n['token']} | {'○' if n['found'] else '✗ 候補'} | {ctx} |")
        lines.append("")
        lines.append("### 固有名詞")
        lines.append("")
        lines.append("| 語 | 原文に存在 |")
        lines.append("|---|---|")
        for t in ch1["terms"]:
            lines.append(f"| {t['token']} | {'○' if t['found'] else '✗ 候補'} |")
        lines.append("")

    # ---- 全章の候補リスト（付録） ----
    lines.append("## 付録: 章ごとのハルシネーション候補（原文に文字列として見つからなかったもの）")
    lines.append("")
    for c in res["chapters"]:
        if not c["has_summary"]:
            continue
        bad_n = [n["token"] for n in c["numbers"] if not n["found"]]
        bad_t = [t["token"] for t in c["terms"] if not t["found"]]
        lines.append(f"### {c['title']}")
        lines.append(f"- 数字の候補: {('、'.join(bad_n) if bad_n else 'なし')}")
        lines.append(f"- 固有名詞の候補: {('、'.join(bad_t) if bad_t else 'なし')}")
        lines.append("")
    for s in res["whole_book_sections"]:
        bad_n = [n["token"] for n in s["numbers"] if not n["found"]]
        bad_t = [t["token"] for t in s["terms"] if not t["found"]]
        lines.append(f"### {s['heading']}（本全体と照合）")
        lines.append(f"- 数字の候補: {('、'.join(bad_n) if bad_n else 'なし')}")
        lines.append(f"- 固有名詞の候補: {('、'.join(bad_t) if bad_t else 'なし')}")
        lines.append("")

    lines.append("## この物差しの限界（v1）")
    lines.append("")
    lines.append("- 原文はOCRテキストで、誤読（例:「ゼロガら」）が混ざる。原文側に誤読があると、正しい要約語でも未一致になる。")
    lines.append("- 表記ゆれに弱い。「50社以上」のような助詞つき表現、漢数字（「百のツボ」等）、丸めた数字は拾い切れない／照合できないことがある。")
    lines.append("- 固有名詞の抽出はカタカナ語と短い鉤括弧用語のみ。漢字の人名・組織名（守島基博など）は v1 では対象外。")
    lines.append("- 「原文に存在する」ことは「その文脈で正しく使われている」ことと同義ではない。意味的な正しさは v2 以降（LLM照合や人手確認）の領分。")
    lines.append("- 未一致はすべて「候補」。確定ハルシネーションかどうかは、候補を原文と見比べて人が判断する前提の設計。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="文書要約エージェント評価キット v1")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("results_v1.md"))
    ap.add_argument("--json", type=Path, default=Path(__file__).with_name("results_v1.json"))
    args = ap.parse_args()

    res = evaluate(args.data_dir)
    args.out.write_text(render_report(res), encoding="utf-8")
    args.json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    m = res["meta"]
    print(f"chapters covered: {m['covered_chapters']}/{m['total_chapters']}")
    print(f"report -> {args.out}")
    print(f"json   -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
