"""마크다운 보고서를 HTML로 바꾼다 (보고서 원문의 숫자를 그대로 옮기고 새로 계산하지 않는다).

실행 (drone_data_lab 폴더):  python scripts/md_to_html.py reports/분석보고서.md
출력: 같은 이름의 .html (reports/분석보고서.html)
지원하는 마크다운: # ## ### 제목, | 표 |, - 목록, **굵게**, `코드`, 문단. 보고서에서 쓰는 만큼만 다룬다.
"""
import html
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CSS = """
  :root {
    --page: #f1f1ee; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --grid: #e1e0d9; --axis: #c3c2b7;
    --ring: rgba(11,11,11,.12); --tint: #edf3fb; --accent: #1f5fae; --code: #eeeeea;
    --font: "Pretendard", "Noto Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --grid: #2c2c2a; --axis: #383835;
      --ring: rgba(255,255,255,.12); --tint: #1c2735; --accent: #6da7ec; --code: #262624; color-scheme: dark;
    }
  }
  :root[data-theme="dark"] {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --grid: #2c2c2a; --axis: #383835;
    --ring: rgba(255,255,255,.12); --tint: #1c2735; --accent: #6da7ec; --code: #262624; color-scheme: dark;
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  html { scroll-behavior: smooth; scroll-padding-top: 16px; }
  body {
    margin: 0; background: var(--page); color: var(--ink); font-family: var(--font); font-size: 16px; line-height: 1.7;
    word-break: keep-all; overflow-wrap: anywhere;
    padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
    padding-block: 28px 56px;
  }
  .sheet { max-width: 960px; margin: 0 auto; background: var(--surface); border: 1px solid var(--ring); border-radius: 4px; padding: 48px 56px 40px; }
  @media (max-width: 700px) { .sheet { padding: 28px 18px; } }
  .kicker { font-size: 14px; font-weight: 600; color: var(--accent); letter-spacing: .02em; }
  h1 { margin: 6px 0 16px; font-size: clamp(26px, 4.4vw, 34px); line-height: 1.3; font-weight: 700; text-wrap: balance; }
  h2 { margin: 40px 0 10px; font-size: 22px; line-height: 1.4; font-weight: 700; padding-top: 8px; border-top: 1px solid var(--axis); }
  h2 .no { color: var(--accent); margin-right: 8px; }
  h3 { margin: 26px 0 8px; font-size: 17px; font-weight: 700; }
  p { margin: 8px 0; }
  ul { margin: 8px 0; padding-left: 20px; }
  li { margin: 4px 0; }
  ul.meta { list-style: none; padding: 12px 0; margin: 0; border-top: 1px solid var(--axis); border-bottom: 1px solid var(--grid); font-size: 14px; color: var(--ink2); }
  ul.meta li { margin: 2px 0; }
  .src { font-size: 13px; color: var(--ink2); list-style: none; margin-left: -20px; }
  .lbl { font-weight: 700; margin-top: 18px; }
  code { font-family: Consolas, "Cascadia Mono", monospace; font-size: .88em; background: var(--code); padding: 1px 5px; border-radius: 4px; }
  .scroll { overflow-x: auto; margin: 10px 0 14px; border: 1px solid var(--grid); border-radius: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 14.5px; min-width: 520px; }
  th { text-align: left; font-size: 13px; font-weight: 600; color: var(--ink2); padding: 8px 10px; background: var(--tint); border-bottom: 1px solid var(--axis); white-space: nowrap; }
  td { padding: 8px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }
  tr:last-child td { border-bottom: 0; }
  td:first-child, th:first-child { white-space: nowrap; overflow-wrap: normal; }
  th.n, td.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  tr.pick td { background: var(--tint); font-weight: 600; }
  nav.toc { margin: 18px 0 4px; padding: 12px 16px; background: var(--tint); border-radius: 6px; font-size: 14px; }
  nav.toc b { display: block; margin-bottom: 4px; color: var(--ink2); font-size: 13px; }
  nav.toc ol { margin: 0; padding-left: 20px; columns: 2; column-gap: 28px; }
  nav.toc a { color: var(--ink); text-decoration: none; }
  nav.toc a:hover { color: var(--accent); text-decoration: underline; }
  @media (max-width: 700px) { nav.toc ol { columns: 1; } }
  a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  @media print {
    @page { size: A4; margin: 12mm; }
    body { background: #fff; padding: 0; font-size: 11.5px; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .sheet { border: 0; padding: 0; max-width: none; }
    h2 { break-after: avoid; margin-top: 22px; font-size: 15px; }
    h3 { break-after: avoid; }
    table { font-size: 10px; min-width: 0; } tr { break-inside: avoid; }
    .scroll { overflow: visible; } nav.toc { display: none; }
  }
"""

NUMERIC = re.compile(r"^[\s\d.,%~+\-−()/건대회개명]*$")


def inline(text):
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    return t


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_html(rows):
    head, body = split_row(rows[0]), [split_row(r) for r in rows[2:]]
    num = [all(r[i] != "" and NUMERIC.match(r[i]) for r in body if i < len(r)) and len(body) > 0 for i in range(len(head))]
    out = ['<div class="scroll"><table><thead><tr>']
    out += [f'<th class="{"n" if num[i] else ""}">{inline(h)}</th>' for i, h in enumerate(head)]
    out.append("</tr></thead><tbody>")
    for r in body:
        pick = any(c == "선택" or c.endswith("(선택)") for c in r)
        out.append(f'<tr{" class=\"pick\"" if pick else ""}>' + "".join(
            f'<td class="{"n" if i < len(num) and num[i] else ""}">{inline(c)}</td>' for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def convert(md_text):
    lines, out, toc = md_text.splitlines(), [], []
    title, i, sec, in_meta = "", 0, 0, True
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s.startswith("# "):
            title = s[2:].strip()
            out.append('<div class="kicker">분석 보고서 · 드론 예측정비</div>')
            out.append(f"<h1>{inline(title)}</h1>")
            i += 1
        elif s.startswith("## "):
            in_meta = False
            sec += 1
            m = re.match(r"(\d+)\.\s*(.*)", s[3:].strip())
            label = m.group(2) if m else s[3:].strip()
            toc.append((sec, label))
            if sec == 1:
                out.append("<!--TOC-->")                      # 목차는 첫 절 앞에 둔다
            out.append(f'<h2 id="s{sec}"><span class="no">{m.group(1) if m else sec}</span>{inline(label)}</h2>')
            i += 1
        elif s.startswith("### "):
            out.append(f"<h3>{inline(s[4:].strip())}</h3>")
            i += 1
        elif s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(table_html(rows))
        elif s.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:])
                i += 1
            cls = ' class="meta"' if in_meta else ""
            lis = "".join(f'<li class="src">{inline(x)}</li>' if x.startswith("출처:") else f"<li>{inline(x)}</li>" for x in items)
            out.append(f"<ul{cls}>{lis}</ul>")
        elif re.match(r"^\d+\.\s", s):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s", "", lines[i].strip()))
                i += 1
            out.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ol>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||- |\d+\.\s)", lines[i].strip()):
                para.append(lines[i].strip())
                i += 1
            text = " ".join(para)
            cls = ' class="lbl"' if re.fullmatch(r"\*\*[^*]+\*\*", text) else ""
            out.append(f"<p{cls}>{inline(text)}</p>")
    body = "\n".join(out)
    nav = ('<nav class="toc" aria-label="목차"><b>목차</b><ol>' +
           "".join(f'<li><a href="#s{n}">{inline(t)}</a></li>' for n, t in toc) + "</ol></nav>")
    return title, body.replace("<!--TOC-->", nav)


def main():
    if len(sys.argv) < 2:
        sys.exit("사용법: python scripts/md_to_html.py reports/분석보고서.md")
    src = Path(sys.argv[1])
    title, body = convert(src.read_text(encoding="utf-8"))
    page = (f'<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{html.escape(title)}</title>\n'
            f"<style>{CSS}</style>\n</head>\n<body>\n<article class=\"sheet\">\n{body}\n</article>\n</body>\n</html>\n")
    out = src.with_suffix(".html")
    out.write_text(page, encoding="utf-8")
    print(f"저장: {out}  ({len(page):,}자, 표 {page.count('<table>')}개, 제목 {page.count('<h2')}개)")


if __name__ == "__main__":
    main()
