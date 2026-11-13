"""Build docs/CivicAI_Project_Document.docx from the course template.

Run from the project root:   python docs/build_document.py
Needs python-docx (see docs/requirements-docs.txt). Word is optional: when it is
installed, the script reads the real page numbers from Word and fills the TOC.

Rules this script follows
- The template is copied, never edited.
- Every result is read from a COMMITTED file (git show HEAD:<path>), never from
  the working folder, so a file that is half-written by another session is not used.
- Every number typed into the text is checked: the script stops if the number
  does not appear in the file named next to it (see the helper `n`).
- The test split is never opened.

Later phases: add the new section text in the "SECTIONS" part at the bottom,
remove the TO BE COMPLETED marker for that section only, and run the script again.
"""

import copy
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "docs" / "screenshots" / "DE-ML Final Project Document Template-1.docx"
OUTPUT = ROOT / "docs" / "CivicAI_Project_Document.docx"
PROJECT_NAME = "CivicAI: Predicting NYC 311 Service Request Resolution"
WORD_PAGES_SCRIPT = ROOT / "docs" / "word_pages.ps1"
TOC_FILE = ROOT / "docs" / "toc_pages.json"          # page numbers read from Word (small helper file)

TEXT_WIDTH_CM = 15.9                                  # A4 minus 2.54 cm margins


# ====================================================================== committed files
def git_show(path):
    """Text of a file as committed in HEAD."""
    return subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True,
                          check=True).stdout.decode("utf-8")


def git_bytes(path):
    return subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


def read_csv(path):
    return pd.read_csv(io.StringIO(git_show(path)))


def notebook(path):
    """(text of all cells and outputs with tags removed, list of html tables per cell)."""
    nb = json.loads(git_show(path))
    text, tables = [], {}
    for i, cell in enumerate(nb["cells"]):
        text.append("".join(cell["source"]))
        for out in cell.get("outputs", []):
            if "text" in out:
                text.append("".join(out["text"]))
            html = out.get("data", {}).get("text/html")
            if html:
                html = "".join(html)
                text.append(re.sub(r"<[^>]+>", " ", html))
                try:
                    table = pd.read_html(io.StringIO(html), index_col=0)[0]
                    if isinstance(table.columns, pd.MultiIndex):
                        table.columns = table.columns.get_level_values(0)
                    tables.setdefault(i, []).append(table)
                except ValueError:
                    pass
    return "\n".join(text), tables


NB01_PATH = "notebooks/01_data_cleaning_eda.ipynb"
NB02_PATH = "notebooks/02_feature_selection.ipynb"
NB03_PATH = "notebooks/03_model_comparison.ipynb"
NB01, T01 = notebook(NB01_PATH)
NB02, T02 = notebook(NB02_PATH)
NB03, T03 = notebook(NB03_PATH)
NB04, T04 = notebook("notebooks/04_stacking.ipynb")
NB05, T05 = notebook("notebooks/05_stacking_improvement.ipynb")
NB06, T06 = notebook("notebooks/06_final_model_xgboost.ipynb")
STACKVAL = read_csv("reports/stacking_validation.csv")
TESTV1 = read_csv("reports/test_results.csv")
EXPS = read_csv("reports/stacking_experiments.csv")
V2VAL = read_csv("reports/stacking_v2_validation.csv")
FWD = read_csv("reports/test_forward_results.csv")
FINALVAL = read_csv("reports/final_model_validation.csv")
META = json.loads(git_show("models/model_meta.json"))
METRICS = json.loads(git_show("models/metrics.json"))
CMP = read_csv("reports/model_comparison.csv")
TUNE = read_csv("reports/tuning_results.csv")
LOG = read_csv("reports/cleaning_log.csv")
SEL = read_csv("reports/feature_selection.csv")
BEST = json.loads(git_show("reports/best_settings.json"))
FETCH = git_show("src/fetch_data.py")
SOURCES = {"nb01": NB01, "nb02": NB02, "nb03": NB03, "nb04": NB04, "nb05": NB05, "nb06": NB06, "fetch": FETCH,
           "meta": git_show("models/model_meta.json"), "metrics": git_show("models/metrics.json"),
           "log": LOG.to_csv(index=False), "sel": SEL.to_csv(index=False)}


def n(source, value):
    """Return `value` as text after checking that it appears in the named committed file.

    This is how every number typed into the document is tied to a saved file.
    """
    value = str(value)
    if value not in SOURCES[source]:
        raise SystemExit(f"NUMBER NOT FOUND: {value!r} is not in source '{source}'")
    return value


def chk(source, snippet):
    """Check that a snippet (for example a list of counts) appears in the named committed file."""
    if snippet not in SOURCES[source]:
        raise SystemExit(f"TEXT NOT FOUND: {snippet!r} is not in source '{source}'")


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


# ====================================================================== word helpers
doc = Document(str(TEMPLATE))
counters = {"Figure": 0, "Table": 0}
SCRATCH = Path(tempfile.mkdtemp(prefix="civicai_doc_"))
BODY_FONT = None


def set_cell_shading(cell, fill):
    props = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    props.append(shd)


def add_runs(paragraph, text, size=None, color=None, bold=False):
    """Tiny markup: **bold** (may contain `code`), `code`, ==highlight==."""
    for part in re.split(r"(\*\*.+?\*\*|`.+?`|==.+?==)", text):
        if not part:
            continue
        if part.startswith("**"):
            add_runs(paragraph, part[2:-2], size=size, color=color, bold=True)
            continue
        if part.startswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
            run.font.size = Pt((size or 11) - 1)
        elif part.startswith("=="):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        else:
            run = paragraph.add_run(part)
            if size:
                run.font.size = Pt(size)
        if bold:
            run.bold = True
        if color:
            run.font.color.rgb = RGBColor.from_string(color)


def P(text, after=6, keep_next=False, align=None, size=None, italic=False):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.15
    paragraph.paragraph_format.keep_with_next = keep_next
    if align:
        paragraph.alignment = align
    add_runs(paragraph, text, size=size)
    if italic:
        for run in paragraph.runs:
            run.italic = True
    return paragraph


def H1(number, title, first=False):
    paragraph = doc.add_paragraph(f"{number}. {title.strip()}", style="Heading 1")
    if not first:
        paragraph.paragraph_format.page_break_before = True
    return paragraph


def H2(text):
    return doc.add_paragraph(text, style="Heading 2")


def H3(text):
    return doc.add_paragraph(text, style="Heading 3")


def make_bullets_numbering():
    """Add a bullet list definition to the template's numbering part (it has none we can use)."""
    numbering = doc.part.numbering_part.element
    ids = [int(x.get(qn("w:abstractNumId"))) for x in numbering.findall(qn("w:abstractNum"))] or [0]
    num_ids = [int(x.get(qn("w:numId"))) for x in numbering.findall(qn("w:num"))] or [0]
    abstract_id, num_id = max(ids) + 1, max(num_ids) + 1
    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "hybridMultilevel")
    abstract.append(multi)
    level = OxmlElement("w:lvl")
    level.set(qn("w:ilvl"), "0")
    for tag, val in (("w:start", "1"), ("w:numFmt", "bullet"), ("w:lvlText", "•"), ("w:lvlJc", "left")):
        element = OxmlElement(tag)
        element.set(qn("w:val"), val)
        level.append(element)
    ppr = OxmlElement("w:pPr")
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "567")
    ind.set(qn("w:hanging"), "283")
    ppr.append(ind)
    level.append(ppr)
    abstract.append(level)
    first_num = numbering.find(qn("w:num"))
    if first_num is not None:
        first_num.addprevious(abstract)
    else:
        numbering.append(abstract)
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    ref = OxmlElement("w:abstractNumId")
    ref.set(qn("w:val"), str(abstract_id))
    num.append(ref)
    numbering.append(num)
    return num_id


BULLET_NUM_ID = make_bullets_numbering()


def BUL(items, size=None):
    for item in items:
        paragraph = doc.add_paragraph(style="List Paragraph")
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.paragraph_format.line_spacing = 1.15
        ppr = paragraph._p.get_or_add_pPr()
        numpr = OxmlElement("w:numPr")
        ilvl = OxmlElement("w:ilvl")
        ilvl.set(qn("w:val"), "0")
        numid = OxmlElement("w:numId")
        numid.set(qn("w:val"), str(BULLET_NUM_ID))
        numpr.append(ilvl)
        numpr.append(numid)
        ppr.append(numpr)
        add_runs(paragraph, item, size=size)
    doc.paragraphs[-1].paragraph_format.space_after = Pt(8)


def CODE(command):
    """A command block: monospaced, light grey."""
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.left_indent = Cm(0.4)
    ppr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "EEEEEE")
    ppr.append(shd)
    run = paragraph.add_run(command)
    run.font.name = "Consolas"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    run.font.size = Pt(9)
    return paragraph


def STEP(what, why, check, command=None):
    """The four parts asked for every step: what was done, why, how it was checked, command."""
    P(f"**What was done.** {what}")
    P(f"**Why.** {why}")
    P(f"**How it was checked.** {check}")
    if command:
        P("**Command to reproduce** (project root, virtual environment active):", after=2, keep_next=True)
        CODE(command)


def add_field(paragraph, instruction, cached):
    """A Word field (PAGE, SEQ) with a cached result so it also shows in viewers that do not update fields."""
    def run_with(kind=None, text=None, instr=None):
        run = paragraph.add_run()
        if kind:
            element = OxmlElement("w:fldChar")
            element.set(qn("w:fldCharType"), kind)
            run._r.append(element)
        if instr:
            element = OxmlElement("w:instrText")
            element.set(qn("xml:space"), "preserve")
            element.text = instr
            run._r.append(element)
        if text:
            run.text = text
        return run
    run_with(kind="begin")
    run_with(instr=f" {instruction} ")
    run_with(kind="separate")
    shown = run_with(text=str(cached))
    run_with(kind="end")
    return shown


def caption_style():
    try:
        return doc.styles["Caption"]
    except KeyError:
        style = doc.styles.add_style("Caption", 1)
        style.base_style = doc.styles["Normal"]
        style.font.size = Pt(9)
        style.font.italic = True
        style.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        return style


def CAPTION(kind, text, above=False):
    counters[kind] += 1
    caption_style()
    paragraph = doc.add_paragraph(style="Caption")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(10 if not above else 3)
    paragraph.paragraph_format.space_before = Pt(0 if not above else 6)
    paragraph.paragraph_format.keep_with_next = above
    label = paragraph.add_run(f"{kind} ")
    label.bold = True
    field_run = add_field(paragraph, f"SEQ {kind} \\* ARABIC", counters[kind])
    field_run.bold = True
    paragraph.add_run(". ")
    add_runs(paragraph, text, size=9)
    return counters[kind]


FIG_DIR = SCRATCH / "figures"
FIG_DIR.mkdir()


def FIG(name, caption, width_cm=14.5, max_height_cm=15.0):
    """Insert a committed figure and a numbered caption under it."""
    from PIL import Image
    path = FIG_DIR / name
    if not path.exists():
        path.write_bytes(git_bytes(f"reports/figures/{name}"))
    w, h = Image.open(path).size
    width = min(width_cm, max_height_cm * w / h)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.add_run().add_picture(str(path), width=Cm(width))
    CAPTION("Figure", caption)


def FIG_GRID(names_and_captions, per_row=3, width_cm=5.0, caption=""):
    """A small grid of figures (one caption for the grid)."""
    rows = [names_and_captions[i:i + per_row] for i in range(0, len(names_and_captions), per_row)]
    table = doc.add_table(rows=len(rows), cols=per_row)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, row in enumerate(rows):
        for c, (name, label) in enumerate(row):
            path = FIG_DIR / name
            if not path.exists():
                path.write_bytes(git_bytes(f"reports/figures/{name}"))
            cell = table.cell(r, c)
            cell.width = Cm(TEXT_WIDTH_CM / per_row)
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.keep_with_next = True
            paragraph.add_run().add_picture(str(path), width=Cm(width_cm))
            small = cell.add_paragraph()
            small.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = small.add_run(label)
            run.font.size = Pt(8)
            run.bold = True
    for row in table.rows:
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
    CAPTION("Figure", caption)


def TABLE(headers, rows, widths, caption, font=9, align=None, bold_rows=(), caption_above=True):
    """A results table with a header row, light shading and a numbered caption above."""
    assert abs(sum(widths) - TEXT_WIDTH_CM) < 0.15, f"widths sum to {sum(widths):.2f}, expected {TEXT_WIDTH_CM}"
    if caption_above:
        CAPTION("Table", caption, above=True)
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = doc.styles["Table Grid"]
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    align = align or ["l"] * len(headers)

    def fill(cell, text, bold=False, alignment="l"):
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.paragraph_format.space_after = Pt(1)
        paragraph.paragraph_format.space_before = Pt(1)
        paragraph.alignment = {"l": WD_ALIGN_PARAGRAPH.LEFT, "r": WD_ALIGN_PARAGRAPH.RIGHT,
                               "c": WD_ALIGN_PARAGRAPH.CENTER}[alignment]
        add_runs(paragraph, str(text), size=font)
        if bold:
            for run in paragraph.runs:
                run.bold = True

    for c, head in enumerate(headers):
        cell = table.rows[0].cells[c]
        fill(cell, head, bold=True, alignment="c" if align[c] != "l" else "l")
        set_cell_shading(cell, "D9E2F3")
    header_props = table.rows[0]._tr.get_or_add_trPr()
    header_props.append(OxmlElement("w:tblHeader"))
    for index, row in enumerate(rows):
        cells = table.add_row().cells
        for c, value in enumerate(row):
            fill(cells[c], value, bold=index in bold_rows, alignment=align[c])
        row_props = table.rows[-1]._tr.get_or_add_trPr()
        row_props.append(OxmlElement("w:cantSplit"))
    for row in table.rows:
        for c, width in enumerate(widths):
            row.cells[c].width = Cm(width)
    if len(rows) <= 14:                                   # small table: keep it on one page
        for row in table.rows[:-1]:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.keep_with_next = True
    grid = table._tbl.tblGrid
    for c, col in enumerate(grid.findall(qn("w:gridCol"))):
        col.set(qn("w:w"), str(int(widths[c] * 567)))
    if not caption_above:
        CAPTION("Table", caption)
    spacer = doc.add_paragraph()                      # a very small gap after the table (cannot spill onto a new page)
    spacer.paragraph_format.space_after = Pt(0)
    spacer.paragraph_format.line_spacing = Pt(6)
    spacer.add_run(" ").font.size = Pt(2)
    return table


def PENDING(phase, evidence, placeholders=None):
    """The highlighted marker for a section that is not finished yet. No results are written."""
    P(f"==TO BE COMPLETED (Phase {phase})==", after=6)
    P("This section will hold:", after=3, keep_next=True)
    BUL(evidence)
    for label in placeholders or []:
        box = doc.add_table(rows=1, cols=1)
        box.style = doc.styles["Table Grid"]
        box.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = box.rows[0].cells[0]
        cell.width = Cm(TEXT_WIDTH_CM - 2)
        set_cell_shading(cell, "F5F5F5")
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(18)
        paragraph.paragraph_format.space_after = Pt(18)
        run = paragraph.add_run(f"[Screenshot placeholder: {label}]")
        run.italic = True
        run.font.size = Pt(10)
        row_props = box.rows[0]._tr.get_or_add_trPr()
        row_props.append(OxmlElement("w:cantSplit"))
        gap = doc.add_paragraph()
        gap.paragraph_format.space_after = Pt(2)


def TODO(text):
    P(f"==TODO== {text}")


# ====================================================================== template edits
def fill_cover():
    """Set PROJECT NAME. MEMBER NAME and every other field stay exactly as in the template."""
    for paragraph in doc.paragraphs:
        if paragraph.text.strip() == "PROJECT NAME":
            paragraph.runs[0].text = PROJECT_NAME
            for run in paragraph.runs[1:]:
                run.text = ""
            return
    raise SystemExit("PROJECT NAME not found on the cover page")


def toc_titles():
    table = doc.tables[0]
    return [(row.cells[0].text.strip(), row.cells[1].text.strip()) for row in table.rows]


def fill_toc_pages(pages):
    """Write the page column of the TOC (rows after the first). `pages` maps section number to page."""
    table = doc.tables[0]
    model_run = table.rows[0].cells[2].paragraphs[0].runs[0]       # row 1 already holds "1": copy its look
    for row in table.rows[1:]:
        number = row.cells[0].text.strip()
        if number in pages:
            cell = row.cells[2]
            paragraph = cell.paragraphs[0]
            for run in list(paragraph.runs):
                run._r.getparent().remove(run._r)
            new_run = paragraph.add_run(str(pages[number]))
            if model_run._r.rPr is not None:
                new_run._r.insert(0, copy.deepcopy(model_run._r.rPr))


def start_body_section():
    """Section break after the TOC; the body gets its own footer with page numbers from 1."""
    doc.add_section(WD_SECTION.NEW_PAGE)
    section = doc.sections[-1]
    section.footer.is_linked_to_previous = False
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in list(footer.runs):
        run._r.getparent().remove(run._r)
    shown = add_field(footer, "PAGE", 1)
    shown.font.size = Pt(10)
    sect_pr = section._sectPr
    page_number = OxmlElement("w:pgNumType")
    page_number.set(qn("w:start"), "1")
    sect_pr.find(qn("w:pgMar")).addnext(page_number)


# ====================================================================== column table data
def short(text, limit=110):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        first = text.split(". ")[0]
        text = first if len(first) <= limit + 30 else text[:limit].rsplit(" ", 1)[0] + " ..."
    return text.rstrip(".") + "."


def column_rows():
    """One row per raw column: name, meaning, type, null %, placeholder %, role, reason."""
    metadata = json.loads((ROOT / "docs" / "dataset_columns.json").read_text(encoding="utf-8"))
    meanings = {c[0]: (c[1], c[2]) for c in metadata}
    missing = T01[37][0]                                             # null % and placeholder % per column
    rules = T01[40][0]
    final_raw = ["complaint_type", "descriptor", "location_type", "open_data_channel_type",
                 "borough", "incident_zip", "latitude", "longitude"]
    selection = SEL.set_index("feature")
    too_empty = re.search(r"Dropped 10 columns: ([^\"]+)", LOG.to_csv(index=False)).group(1).split(", ")
    leak = ["status", "resolution_description", "resolution_action_updated_date", "due_date"]
    published = {"calendar_date": "date", "text": "text", "number": "number", "point": "point"}
    rows = []
    for name in meanings:
        kind, description = meanings[name]
        if name in final_raw:
            role, reason = "Feature", selection.loc[name, "reason"]
        elif name == "created_date":
            role, reason = "Feature source", "Source of hour, weekday, month and is_weekend. With closed_date it also gives the target."
        elif name == "closed_date":
            role, reason = "Target source", "Used once to compute the target, then dropped (leakage). The 702 rows without it are removed."
        elif name == "unique_key":
            role, reason = "Id only", "Request id. Used to find duplicates and to sort. Never a feature."
        elif name in ("agency", "address_type", "community_board"):
            role, reason = "Dropped (Phase 4)", selection.loc[name, "reason"]
        elif name in leak:
            role = "Dropped (leakage)"
            reason = rules.loc[name, "reason"] if name in rules.index else "Leakage: only known after the request is handled."
            if name == "due_date":
                reason = "Leakage: a later date. Also more than 95% empty."
        elif name in too_empty:
            role, reason = "Dropped", "More than 95% empty in the training period."
            if name in ("facility_type", "park_facility_name"):
                reason += " Mostly placeholder text."
        else:
            role, reason = "Dropped", rules.loc[name, "reason"]
        reason = reason.replace("Free-text street address with thousands of different values; location is already covered by ZIP, borough and coordinates",
                                "Free-text address with thousands of values. ZIP, borough and coordinates already give the location")
        rows.append([f"`{name}`", short(description), published.get(kind, kind),
                     f"{float(str(missing.loc[name, 'null_pct']).rstrip('%')):.2f}",
                     f"{float(str(missing.loc[name, 'placeholder_pct']).rstrip('%')):.2f}",
                     role, reason])
    return rows


# ====================================================================== SECTIONS
def section_1():
    problem_days = n("nb01", "11.94")
    P("New York City has a phone and web service, 311, where people report problems that are not emergencies: noise, "
      "a broken heater, a blocked driveway, a rat sighting. Each report is a **service request**. A city agency "
      "handles it and then closes it.")
    P(f"Some requests are closed in hours. Others stay open for months. In the training period of this project the "
      f"median request was closed in {n('nb01', '7.0')} hours, but the average was {problem_days} days, because a "
      f"small number of requests stay open for a very long time (notebook 01, section 1.4).")
    H2("1.1 The problem")
    P("When someone files a request, they and the agency staff would like to know one thing: **will it be resolved "
      "within a week?** A yes/no answer helps staff find the requests that are likely to be slow and helps the "
      "person set expectations.")
    P("This project builds a **classifier** that answers this question from the information known when the "
      "request is filed. The answer is a probability and a class: 1 = resolved within 7 days, 0 = took longer.")
    H2("1.2 Why 311 data")
    BUL([f"**It is open and public.** The data comes from NYC Open Data, dataset \"311 Service Requests from 2020 to "
         f"Present\" (id `erm2-nwe9`). Anyone can download it with the Socrata API.",
         f"**It is large.** For the date range used here (2024-01-01 to 2026-08-31) the full table holds "
         f"{n('nb01', '9,581,154')} closed requests, so a fair sample is easy to take.",
         "**It records both dates.** Every closed request has a `created_date` and a `closed_date`. Their difference "
         "gives the correct answer for past requests, so no one has to label data by hand.",
         "**It mixes kinds of information.** There are categories (complaint type, borough), a time stamp and a "
         "location. This is a good test for data cleaning, feature selection and model comparison."])
    H2("1.3 What the system does")
    P("The finished system has four parts. A person (or another program) sends the raw fields of a new request to a "
      "web service. The service cleans the fields with the same code that was used in training, runs the saved "
      "model, and returns the prediction.")
    BUL(["**Data and cleaning** (sections 3 and 4). A sample of closed requests is pulled, checked and cleaned.",
         "**Models** (sections 5 and 6). Six candidate models are compared and the best three are chosen.",
         "**Meta-classifier and final model** (section 7). The models are combined with a stacking classifier, which is compared with its base models. "
         "The final model, XGBoost alone, is then chosen and explained.",
         "**Deployment** (sections 8 to 10). A FastAPI service on an AWS EC2 server returns predictions and is tested "
         "end to end."])
    H2("1.4 How this document is organised")
    P("The sections follow the course template. Sections 1 to 7 describe finished work (Phases 0 to 6c of the project "
      "plan). Sections 8 to 11 are marked **TO BE COMPLETED** until their phase is finished. Section 12 is written now.")
    TABLE(["Section", "Title", "Phase", "Status"],
          [["1", "Introduction", "0", "Filled"], ["2", "Objective", "2", "Filled"],
           ["3", "Dataset Column Details", "1, 2, 4", "Filled"], ["4", "Data Cleaning and Graph Plot", "3, 4", "Filled"],
           ["5", "Model Selection and Train Each Model", "5", "Filled"], ["6", "Selecting best 3 Models", "5", "Filled"],
           ["7", "Use Meta Classifier and Create final model", "6, 6b, 6c", "Filled"],
           ["8", "Create EC2 instance", "8", "To be completed"], ["9", "Deploy Model on EC2", "8", "To be completed"],
           ["10", "Test Model using Postman (API Call)", "9", "To be completed"],
           ["11", "Colab Link", "10", "To be completed"], ["12", "Future Scope", "-", "Filled"]],
          [2.0, 8.0, 2.4, 3.5], "Sections of this document and their status.", font=9, align=["c", "l", "c", "l"])
    P("**How to reproduce the work.** Every step below gives a command. Run them from the project root with the "
      "virtual environment active (`venv\\Scripts\\activate` on Windows). Notebooks are run with "
      "`jupyter nbconvert --to notebook --execute --inplace <notebook>`. All random seeds are 42.")


def section_2():
    H2("2.1 The target variable")
    P("The model predicts **`resolved_within_7d`**.")
    BUL(["It is **1** if `closed_date` minus `created_date` is 7 days (168 hours) or less.",
         "It is **0** if the request took longer than 7 days.",
         "It is computed once, in `add_target` in `src/preprocessing.py`. After that, `closed_date` is dropped and is "
         "never a model input."])
    H2("2.2 Why 7 days: the class-balance check")
    STEP("Durations (`closed_date` minus `created_date`, in hours) were computed on the **training period** only "
         f"({n('nb01', '139,206')} rows, 2024-01-01 to 2025-12-02). For each possible threshold, the share of requests "
         "closed within it was counted.",
         "A threshold must be easy to explain and must leave enough requests in the smaller class. The model learns "
         "little if one class is very small. Using only the training period stops information from the validation "
         "and test periods from leaking into the choice.",
         "The project rule says: if the smaller class is under 20% of the rows, ask the student before going on. "
         "The check gave 17.2%, so the question was asked and **the student confirmed 7 days**.",
         "jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_cleaning_eda.ipynb")
    t9 = T01[9][0]
    rows = [[str(int(day)), t9.loc[day, "resolved_within"], t9.loc[day, "not_resolved_within"]] for day in t9.index]
    rows = [[f"{r[0]} days" if r[0] != "1" else "1 day", r[1], r[2]] for r in rows]
    TABLE(["Threshold", "Class 1: resolved within", "Class 0: took longer"], rows, [4.0, 5.9, 6.0],
          "Share of training requests in each class for each possible threshold (notebook 01, section 1.4).",
          align=["l", "r", "r"], bold_rows=[4])
    FIG("01_positive_share_by_threshold.png", "Share resolved within each candidate threshold. The 7-day bar is highlighted.", 12.0)
    P(f"At 7 days there are {n('nb01', '115,277')} class 1 requests (82.8%) and {n('nb01', '23,929')} class 0 requests "
      f"(17.2%) in the training period. The smaller class is {n('nb01', '23,929')} rows, which is plenty to learn from.")
    FIG("03_class_balance.png", "Class balance of the training period for the 7-day target.", 9.5)
    chk("nb01", "| 5 days | 79.8% | 20.2% |")
    chk("nb01", "| 3 days | 74.6% | 25.4% |")
    chk("nb01", "| Median (7.0 hours) | 50% | 50% |")
    P("**Other thresholds that were considered** (notebook 01, section 1.8):")
    TABLE(["Threshold", "Class 1", "Class 0", "Comment"],
          [["7 days (chosen)", "82.8%", "17.2%", "Easy to explain (\"within a week\"). Slightly under the 20% guide."],
           ["5 days", "79.8%", "20.2%", "Passes the guide by 0.2 points only. A less natural cut-off."],
           ["3 days", "74.6%", "25.4%", "Better balance. The proposed alternative."],
           ["Median (7.0 hours)", "50%", "50%", "Perfect balance, but it changes the question to \"closed the same day?\"."]],
          [3.4, 1.8, 1.8, 8.9], "Options for the threshold. All shares are from the training period.",
          align=["l", "r", "r", "l"], bold_rows=[0])
    P("**Why 7 days was kept.** The imbalance is mild (about 1 row in 6). It is handled without changing the "
      "question: the models use class weights, and they are judged with **macro F1 and ROC-AUC**, not accuracy alone. "
      "A model that always answers \"within 7 days\" is already right for 83.2% of validation requests "
      f"(accuracy {CMP.loc[0, 'accuracy']:.3f} in `reports/model_comparison.csv`), so accuracy alone would be misleading.")
    P(f"**The target is stable over time.** Every month of the sample is between {n('nb01', '78.9%')} and "
      f"{n('nb01', '86.0%')} resolved within 7 days, so one threshold works for the whole date range.")
    FIG("01_positive_share_by_month.png", "Share resolved within 7 days by month of filing (all months, used as a check only).", 14.5)
    P("**Limits to remember.** The sample holds **closed** requests only. In the newest months, slow requests that "
      "are still open are missing, which can push the recent share up a little. Also, \"closed\" means the agency "
      "closed the request in its system. It does not always mean the problem was physically fixed.")
    FIG("01_duration_histogram_log.png", "How long requests take to close (log scale on the x axis, training period).", 13.0)
    H2("2.3 Course requirements met by the project")
    TABLE(["#", "Course requirement", "Where in this document", "Status"],
          [["1", "Data cleaning and preprocessing, with charts and stated decisions", "Sections 3 and 4", "Filled"],
           ["2", "Feature selection, with the target and a reason for each feature", "Sections 2 and 3", "Filled"],
           ["3", "Model selection, training and testing, with a metric comparison", "Sections 5 and 6", "Filled"],
           ["4", "A meta-classifier (stacking) compared with its base models", "Section 7", "Filled"],
           ["5", "Deployment on EC2 with an API, tested end to end", "Sections 8 to 10", "To be completed"]],
          [0.8, 8.2, 4.0, 2.9], "The five course requirements and where each one is covered.", align=["c", "l", "l", "l"])


def section_3():
    H2("3.1 Source and sample design")
    STEP(f"The data was pulled from the Socrata API of NYC Open Data (`https://data.cityofnewyork.us/resource/erm2-nwe9.json`, "
         f"parameters `$where`, `$limit`, `$offset`, `$order`). Only requests with status **Closed** and a `created_date` from "
         f"{n('nb01', '2024-01-01')} to {n('nb01', '2026-08-31')} were requested. The script keeps every request whose "
         f"`unique_key` divides evenly by {n('fetch', '48')}, which is a **1-in-48 sample**. The pull is split into "
         f"{n('fetch', '4')} date windows per month. Each window is saved as its own file, so an interrupted run can resume, and "
         f"a failed call is retried up to {n('fetch', '5')} times. The pieces are joined into `data/raw/nyc311_raw.parquet` "
         f"without changing any value. All columns are stored as text.",
         f"The full table has {n('nb01', '9,581,154')} closed requests in this range. The assignment needs 150,000 to 300,000 "
         "rows. `unique_key` is only an id number, so keeping every 48th id gives a fair sample of every day and hour, "
         "and the same rows come back on every run.",
         f"The raw file has **{n('nb01', '199,580')} rows and {n('nb01', '44')} columns**. The earliest request is "
         f"{n('nb01', '2024-01-01 00:00:00')} and the latest is {n('nb01', '2026-08-31 23:59:06')}. No date is outside the "
         "range and no day is without requests (notebook 01, section 2.5). The next section compares the sample with the full table.",
         "python src/fetch_data.py")
    H3("How I know the sample matches the full table")
    P(f"The notebook asks the API to count **all** closed requests in the same date range, by borough and by complaint "
      f"type, and compares the shares with the sample (notebook 01, section 1.7). The largest difference in a borough "
      f"share is {n('nb01', '0.12')} percentage points. The largest difference in a complaint-type share is "
      f"{n('nb01', '0.05')} percentage points. The sample's ten most common complaint types are the same as the full "
      f"table's ten most common.")
    t20 = T01[20][0]
    rows = [[str(b), f"{int(t20.loc[b, 'full_rows']):,}", f"{int(t20.loc[b, 'sample_rows']):,}",
             str(t20.loc[b, "full_share"]), str(t20.loc[b, "sample_share"]), f"{float(t20.loc[b, 'diff_pct_points']):+.2f}"]
            for b in t20.index]
    TABLE(["Borough", "Full table rows", "Sample rows", "Full share", "Sample share", "Difference (points)"], rows,
          [3.2, 2.9, 2.5, 2.2, 2.4, 2.7], "Borough shares: full table against the sample (notebook 01, section 1.7).",
          align=["l", "r", "r", "r", "r", "r"])
    H3("Share of requests that were not closed")
    TODO("The share of requests that were **not closed** is not saved in a committed file. The script `src/fetch_data.py` "
         "asks the API for the number of requests per status and saves it to `data/raw/status_counts.csv`, but `data/raw/` is "
         "not committed. Commit that small file (or copy its numbers into a committed report) and this paragraph can be filled in.")
    P(f"What is known from committed files: in the sample, {n('nb01', '702')} rows ({n('nb01', '0.35%')}) have status Closed "
      "but an empty `closed_date`. They have no duration, so they are removed in section 4.")
    H2("3.2 Column details (all 44 raw columns)")
    P("The table lists every raw column. **Meaning** is taken from the dataset's published metadata "
      "(`https://data.cityofnewyork.us/api/views/erm2-nwe9.json`), shortened. **Type** is the published type. The raw file "
      "stores every column as text. The types used after cleaning are in section 4.1. **Null %** is the share of empty cells. "
      "**Placeholder %** is the share of cells holding text such as Unspecified or N/A, which also means \"no value\". "
      "Both are measured on all 199,580 raw rows (notebook 01, section 2.3).")
    TABLE(["Column", "Meaning", "Type", "Null %", "Place-holder %", "Role", "Reason"], column_rows(),
          [2.9, 3.6, 1.3, 1.0, 1.1, 1.8, 4.2], "All 44 raw columns, with role and reason.", font=7.5,
          align=["l", "l", "l", "r", "r", "l", "l"])
    P("Roles: **Feature** = used by the model. **Feature source** = the four time features are made from it. "
      "**Target source** = used only to compute the target. **Id only** = used to find duplicates, never a feature. "
      "**Dropped** = not used (reason given). Cleaning decisions for each column are in section 4.")
    H2("3.3 Target and feature selection")
    P("The target is described in section 2. This subsection explains which columns the model may use and why.")
    H3("Leakage rules")
    P("**Leakage** means giving the model information that does not exist when a new request is filed. Such a model "
      "looks perfect in testing and is useless in real use. These columns are **never** model inputs:")
    BUL(["`closed_date`, `duration_hours` and `resolved_within_7d`: they are the answer.",
         "`status`, `resolution_description`, `resolution_action_updated_date`, `due_date`: they are written after the request is handled or updated.",
         "`unique_key`: an id with no meaning. `year`: the model would learn about a past period instead of about requests.",
         "`created_date` itself: only its four parts (hour, weekday, month, weekend flag) are used."])
    P("The list is `EXCLUDED_COLUMNS` in `src/features.py`. The function `assert_no_excluded` stops with an error if one of "
      "these columns is among the inputs. It runs every time a pipeline is built and every time it transforms data. "
      "Notebook 02 shows that the check passes, and that it refuses a column that leaks (`duration_hours`).")
    H3("Why the chronological split comes before feature selection")
    P("Choosing features is a form of learning. If it were done on all rows, the validation and test rows would "
      "influence which features are kept, and the test score would be too optimistic. So the data is split **first**, "
      "and every measurement of \"how useful is this feature\" uses the **training split** only.")
    P("The split is by time, not at random. The model will always be used on requests filed **after** it was trained. A "
      "random split would let it learn from requests newer than the ones it is tested on.")
    t4 = T02[4][0]
    rows = [[s, f"{int(t4.loc[s, 'rows']):,}", str(t4.loc[s, "share_of_rows"]), str(t4.loc[s, "from"]), str(t4.loc[s, "to"]),
             str(t4.loc[s, "share_resolved_within_7d"])] for s in t4.index]
    rows = [[{"train": "Train", "val": "Validation", "test": "Test"}[r[0]]] + r[1:] for r in rows]
    TABLE(["Split", "Rows", "Share", "From", "To", "Class 1 share"], rows, [2.0, 1.9, 1.5, 4.0, 4.0, 2.5],
          "Chronological split of the 198,867 cleaned rows (notebook 02, section 2).", font=8,
          align=["l", "r", "r", "l", "l", "r"])
    P("**How it was checked.** Notebook 02 confirms that every training request is older than every validation request, and "
      "every validation request is older than every test request. The test split is **not opened** in the feature selection "
      "notebook. It is scored once, at the end, in Phase 6.")
    CODE("python -m src.features")
    P("This command makes `train.parquet`, `val.parquet` and `test.parquet` in `data/processed/` from `clean.parquet`.", after=8)
    H3("Candidate features")
    P("There were 15 candidates: the 11 cleaned columns (nine text columns plus latitude and longitude) and the four "
      "time features made from `created_date` by `add_time_features`: `hour` (0 to 23), `weekday` (0 = Monday), `month` "
      "(1 to 12) and `is_weekend` (1 = Saturday or Sunday).")
    H3("Evidence used")
    P("Four different measurements were used, because each one has a weakness. All of them are in notebook 02, section 6.")
    CODE("jupyter nbconvert --to notebook --execute --inplace notebooks/02_feature_selection.ipynb")
    P(f"**1. Mutual information.** How much knowing a feature reduces the uncertainty about the target (0% = nothing, "
      f"100% = gives the answer away). Its weakness: it favours columns with many categories. To show how large this effect is, "
      f"the calculation was repeated with the target shuffled. Even for columns with hundreds of values the shuffled result stays "
      f"near zero (at most {n('nb02', '0.18%')}), so the large values are real. `complaint_type` ({n('nb02', '54.11%')}) and "
      f"`descriptor` ({n('nb02', '53.30%')}) dominate.")
    FIG("04_mutual_information.png", "Mutual information between each candidate feature and the target (training split).", 13.0)
    P(f"**2. Permutation importance.** A quick, untuned XGBoost model was trained on the training split. One input column "
      f"at a time was shuffled on the validation split, and the drop in ROC-AUC was recorded. The model relies almost "
      f"entirely on `agency` ({n('nb02', '0.3160')}) and `complaint_type` ({n('nb02', '0.1594')}). Its weakness: when two "
      f"features carry the same information, one looks useless. That is why redundancy was checked next.")
    FIG("04_permutation_importance.png", "Drop in validation ROC-AUC when one input column is shuffled.", 13.0)
    P("**3. Redundancy.** If I know feature A, how often can I name feature B by always guessing the most common B?")
    t25 = T02[25][0]
    rows = [[str(idx), str(t25.iloc[i, 0]), str(t25.iloc[i, 1]),
             str(t25.iloc[i, 2]), str(t25.iloc[i, 3])] for i, idx in enumerate(t25.index)]
    TABLE(["If we know", "We can name", "Correct", "Values known", "Values named"], rows, [3.8, 3.4, 2.4, 3.2, 3.1],
          "Redundancy between features (training split, notebook 02, section 6.3).", font=8.5,
          align=["l", "l", "r", "r", "r"])
    P(f"The ZIP code gives the borough {n('nb02', '99.3%')} of the time. The complaint type gives the agency "
      f"{n('nb02', '99.0%')} of the time. So `agency` is almost a coarser copy of `complaint_type`, and `borough` and "
      f"`community_board` add little once `incident_zip` and the coordinates are present.")
    P("**4. Drop-one-group test.** The quick model was trained again from scratch **without** one group of features, and "
      "the validation scores were compared. A negative change means the group was helping. Very small differences (about "
      "0.001 or less) come from a single trained model and are treated as \"no clear difference\".")
    FIG("04_drop_one_group.png", "Change in validation macro F1 when one feature group is removed.", 13.0)
    P("**A closer look at `agency`.** `agency` is the only doubtful group that really helps, but a person filing a request "
      "does not know which agency will handle it, and the API request has no agency field. Four ways of using it were compared:")
    t31 = T02[31][0]
    rows = [[str(i), f"{float(t31.loc[i, 'roc_auc']):.4f}", f"{float(t31.loc[i, 'macro_f1']):.4f}"] for i in t31.index]
    TABLE(["Setup (validation split)", "ROC-AUC", "Macro F1"], rows, [10.1, 2.9, 2.9],
          "What the model scores when `agency` is supplied, guessed, left empty or not used (notebook 02, section 6.5).",
          align=["l", "r", "r"])
    P(f"The clear benefit exists only when the **true** agency is supplied (macro F1 {n('nb02', '0.8276')}). If the agency is "
      f"left empty, the model trained with it gets much worse ({n('nb02', '0.7424')}). Guessing it from the complaint type "
      f"adds nothing the model cannot get from `complaint_type`. **Decision, confirmed by the student: drop `agency`.** "
      f"The cost is small: macro F1 {n('nb02', '0.8191')} without it.")
    H3("Final features: 12 kept, 3 dropped")
    P("The table gives, for each of the 15 candidates, the decision, the evidence and the reason (`reports/feature_selection.csv`). "
      "**MI %** = mutual information. **AUC drop** = permutation importance. **F1 change** = change in macro F1 when "
      "the group is removed. For latitude, longitude and the four time features the last two values are for the group as a whole.")
    rows = []
    for _, r in SEL.iterrows():
        rows.append([f"`{r['feature']}`", r["decision"], f"{r['mi_pct']:.2f}", f"{r['perm_auc_drop']:.4f}",
                     f"{r['drop_f1_change']:+.4f}", r["reason"], r["can_the_caller_supply_it"]])
    TABLE(["Feature", "Decision", "MI %", "AUC drop", "F1 change", "Reason", "Can the caller supply it?"], rows,
          [2.5, 1.4, 1.0, 1.3, 1.4, 4.5, 3.8], "Kept and dropped features with evidence and reasons.", font=7.5,
          align=["l", "c", "r", "r", "r", "l", "l"])
    kept = ", ".join(f"`{f}`" for f in SEL[SEL["decision"] == "KEEP"]["feature"])
    P(f"**Final features (12):** {kept}. **Dropped (3):** `agency`, `address_type`, `community_board`. None of the three can "
      "be given by a person filing a request. Two of them add nothing measurable. `agency` adds a little, but only when the true "
      "agency is known. Every kept feature is either sent by the caller or worked out by the server from `created_at`, so the "
      "API request format does not change.")


def section_4():
    P("Every problem below has the same four parts: a **check** (how big is it?), a **chart**, the **fix** and the **reason**. "
      "The fixes are functions in `src/preprocessing.py`. The notebook calls them, and the API will call the same ones, so a new "
      "request is cleaned exactly like the training data.")
    P("**Two rules.** (1) Anything learned from the data, such as which categories are too rare to keep, is learned from the "
      f"**training period only** ({n('nb01', '139,206')} rows, the oldest 70%). (2) Columns known only after a request is "
      "handled are never inputs. Types and text are fixed first, because values such as \"Unspecified\" are really missing values.")
    CODE("python src/preprocessing.py\njupyter nbconvert --to notebook --execute --inplace notebooks/01_data_cleaning_eda.ipynb")
    P("The first command rebuilds `data/processed/clean.parquet` and `models/cleaning_values.json` from the raw file.", after=8)
    H2("4.1 Wrong types")
    P(f"**Check.** The raw file stores all {n('nb01', '44')} columns as text. Five columns we need are not text: two dates, "
      "two numbers and a ZIP code. For each one the notebook counts the values that are valid, missing, or present but invalid. "
      "A ZIP code is valid only if it has exactly 5 digits and starts with an NYC prefix. `10000` has an NYC prefix but is not a real ZIP code.")
    t26 = T01[26][0]
    rows = [[f"`{c}`", f"{int(t26.loc[c, 'valid']):,}", f"{int(t26.loc[c, 'missing']):,}", f"{int(t26.loc[c, 'invalid']):,}",
             str(t26.loc[c, "new_type"])] for c in t26.index]
    TABLE(["Column", "Valid", "Missing", "Invalid", "Type after cleaning"], rows, [3.6, 2.6, 2.4, 2.4, 4.9],
          "Valid, missing and invalid values for the columns that need a type change (notebook 01, section 2.1).",
          align=["l", "r", "r", "r", "l"])
    FIG("02_type_conversion.png", "Share of valid, missing and invalid values in the five columns that need a type change.", 14.5)
    P(f"**Fix.** Both dates become datetimes and latitude and longitude become numbers. Every non-empty value was valid, so "
      f"nothing was lost. `incident_zip` stays **text** with 5 digits: a ZIP code is a label, not a quantity. The "
      f"{n('nb01', '36')} invalid ZIP values (most often `10000`, {n('nb01', '29')} times) become missing, not guessed. "
      f"Functions: `convert_dates`, `clean_coordinates`, `clean_zip`.")
    H2("4.2 Inconsistent data")
    P("**Check.** For each text column kept, the notebook looks for three problems: the same value written two ways (case or "
      "spaces), placeholder text that means \"no value\", and invalid ZIP codes (counted in 4.1).")
    t31 = T01[31][0]
    rows = [[f"`{c}`", f"{int(t31.loc[c, 'unique_before']):,}", f"{int(t31.loc[c, 'unique_after']):,}",
             f"{int(t31.loc[c, 'values_merged_by_case']):,}", f"{int(t31.loc[c, 'placeholder_rows']):,}"] for c in t31.index]
    TABLE(["Column", "Distinct values before", "After", "Merged by case", "Placeholder rows"], rows, [4.2, 3.6, 2.2, 2.9, 3.0],
          "Inconsistent text in the kept text columns (notebook 01, section 2.2).", align=["l", "r", "r", "r", "r"])
    FIG("02_placeholder_values.png", "Rows holding placeholder text (or an invalid ZIP) in the kept columns.", 13.0)
    P("**Fix** (`standardize_text`): trim spaces, collapse double spaces, and upper-case every kept text column. Upper case is "
      "used because the borough and many complaint types already are, and because the API can then accept `brooklyn`, "
      "`Brooklyn` or `BROOKLYN` as the same value. Placeholders (`Unspecified`, `N/A`, `UNKNOWN`, and community boards such as "
      "`Unspecified BROOKLYN`) become **`Unknown`**. One label for \"no value\" is clearer than five.")
    P("**A side effect.** `ELEVATOR` (HPD) and `Elevator` (Department of Buildings) become one complaint type, and the same for "
      "`PLUMBING` and `Plumbing`. They come from different agencies. This is accepted because the `agency` column was still kept at "
      "that stage, so the difference was not lost.")
    H2("4.3 Missing values")
    P("**Check.** Percent missing for all 44 raw columns, counting both empty cells and placeholder text (the table in section 3.2). "
      "The decision to drop a column is made on the training period only.")
    FIG("02_missing_by_column.png", "Percent missing (empty or placeholder) for every raw column.", 13.0, 17.0)
    P("**Fix, part 1: drop columns more than 95% empty.** A column that is empty for more than 19 rows in 20 cannot help a "
      "general model. Ten columns were dropped: `taxi_company_borough`, `road_ramp`, `bridge_highway_direction`, `facility_type`, "
      "`park_facility_name`, `due_date`, `bridge_highway_segment`, `bridge_highway_name`, `taxi_pick_up_location` and "
      "`vehicle_type`. Most only apply to taxi or bridge and highway complaints.")
    P("**Fix, part 2: one rule for every remaining column.** There are three rules:")
    BUL(["**Fill with `Unknown`** for text columns we keep. \"The caller did not give this\" is useful information, and a label keeps the row.",
         "**Leave missing** for latitude and longitude. There is no honest value to put in. The model pipeline fills them with a value learned from the training rows and adds a \"was missing\" flag.",
         "**Drop the column** if it leaks the answer, repeats another column, or is a free-text address with thousands of values. In total 21 columns were dropped by this rule, "
         "7 are filled with `Unknown`, 4 have nothing missing and 2 are left missing."])
    chk("nb01", "{'Drop column': 21, \"Fill with 'Unknown'\": 7, 'Keep (nothing missing)': 4, 'Leave missing': 2}")
    H2("4.4 Duplicates")
    P("**Check.** Three levels: the same `unique_key` twice, two rows identical in all 44 columns, and **near-duplicates** "
      "(different `unique_key`, but the same complaint type, the same place and the same `created_date` to the second).")
    t43 = T01[43][0]
    rows = [[str(i), f"{int(t43.loc[i, 'extra_rows']):,}"] for i in t43.index]
    TABLE(["Kind of duplicate", "Extra rows"], rows, [11.0, 4.9], "Duplicates found in the raw sample (notebook 01, section 2.4).",
          align=["l", "r"])
    FIG("02_duplicates.png", "Duplicate checks on the raw sample.", 13.5)
    P(f"**Fix.** Exact duplicates are removed with `drop_duplicates` (on all columns, then on `unique_key`). There are none, so no rows "
      f"are removed. This was expected, because the sample was taken by `unique_key`. **Near-duplicates are kept** "
      f"({n('nb01', '119')} extra rows by address, about 0.1% of rows). Each one has its own `unique_key`, so the city recorded it as a "
      f"separate request. Several people can report the same noisy party. They are too few to change the results, and deleting real "
      f"requests would be a bigger mistake than keeping them.")
    H2("4.5 Outliers")
    P("Four things were checked: very long durations, zero and negative durations, coordinates outside New York City, and unusual `created_date` values.")
    H3("Extreme durations")
    t48 = T01[48][0]
    rows = [[str(i), f"{float(t48.loc[i, 'hours']):,.1f}", f"{float(t48.loc[i, 'days']):,.1f}"] for i in t48.index]
    TABLE(["Point in the distribution", "Hours", "Days"], rows, [7.9, 4.0, 4.0],
          "Distribution of the time to close (all usable rows; notebook 01, section 2.5).", align=["l", "r", "r"])
    FIG("02_duration_boxplot_by_agency.png", "Time to close by agency (the 10 agencies with the most requests, log scale).", 13.5)
    FIG("02_duration_histogram_days.png", "Time to close in days on a normal axis, to show the long tail.", 13.5)
    P(f"**Check result.** By the boxplot rule, durations above {n('nb01', '190 hours')} count as outliers ({n('nb01', '32,221')} rows, "
      f"{n('nb01', '16.2%')}). {n('nb01', '15,237')} requests took longer than 30 days and {n('nb01', '511')} took longer than 365 days.")
    P("**Decision: keep the long durations.** They are real: some building and tree requests do stay open for months. They also "
      "cannot distort the model, because the model predicts only \"within 7 days, yes or no\". A request that took 20 days and "
      "one that took 900 days are both class 0. `duration_hours` stays in the cleaned file for analysis but is never an input.")
    H3("Zero and negative durations")
    t52 = T01[52][0]
    rows = [[str(i), f"{int(t52.loc[i, 'zero_rows']):,}", f"{int(t52.loc[i, 'all_rows']):,}", str(t52.loc[i, "share_of_agency"])]
            for i in t52.index]
    TABLE(["Agency", "Zero-duration rows", "All rows", "Share of the agency"], rows, [4.0, 4.2, 3.7, 4.0],
          "Requests closed at the same second they were created, by agency (notebook 01, section 2.5).",
          align=["l", "r", "r", "r"])
    FIG("02_zero_duration_by_agency.png", "Share of each agency's requests with a zero duration.", 12.0)
    P(f"**Check.** {n('nb01', '3,458')} rows ({n('nb01', '1.74%')}) have a zero duration. Almost all come from four agencies and a few "
      f"complaint types (rodent, water system, street condition, construction), where the agency's system closes the request at the moment it is logged. "
      f"**Decision: keep them.** A new request of the same kind would be recorded the same way, so this is a pattern the model should learn. "
      f"All of them are class 1. {n('nb01', '11')} rows have a **negative** duration (closed before created). That is impossible, so "
      f"they are **dropped**. Nine of them are negative by less than a minute.")
    chk("nb01", "Negative by less than one minute: 9 of 11")
    H3("Coordinates outside New York City")
    P(f"**Check.** New York City fits inside latitude {n('nb01', '40.47')} to {n('nb01', '40.93')} and longitude -74.27 to -73.68. "
      f"The notebook found {n('nb01', '196,284')} rows with coordinates, {n('nb01', '3,296')} without, and **0** outside the box. "
      "**Fix.** Nothing changes in this dataset. The rule stays in `clean_coordinates` (a point outside the box becomes missing), "
      "because the API will receive coordinates typed by users.")
    FIG("02_coordinates_map.png", "Request locations (latitude and longitude) inside the New York City box.", 9.5, 12.0)
    H3("Unusual dates")
    P(f"**Check.** Every `created_date` is inside the requested range and every day has requests. Requests per full week range from "
      f"{n('nb01', '1,129')} to {n('nb01', '2,200')} (median {n('nb01', '1,419')}). {n('nb01', '134')} rows ({n('nb01', '0.07%')}) are stamped exactly "
      "00:00:00, almost all from the Health Department, so their true hour was probably not recorded. **Decision.** Nothing to fix. "
      "These rows are kept because they are too few to affect the hour feature.")
    FIG("02_created_per_week.png", "Requests created per week.", 14.0)
    H2("4.6 Rare categories")
    P(f"**Check.** Some text columns have many values that appear only a few times. A model cannot learn from a category with three "
      f"examples. **Fix.** In each text column, a category with fewer than {n('nb01', '50')} rows **in the training period** is replaced by "
      f"`Other`. The list of kept categories is saved in `models/cleaning_values.json`. It is learned from training rows only. "
      f"A category that first appears later, or one the API has never seen, also becomes `Other`, which is what will happen in real use.")
    t64 = T01[64][0]
    rows = [[f"`{c}`", f"{int(t64.loc[c, 'categories_in_train']):,}", f"{int(t64.loc[c, 'kept']):,}", f"{int(t64.loc[c, 'become_Other']):,}",
             str(t64.loc[c, "pct_rows_Other"])] for c in t64.index]
    TABLE(["Column", "Categories in training", "Kept", "Become Other", "Rows made Other"], rows, [4.2, 3.6, 2.2, 2.9, 3.0],
          "Effect of the 50-row rule (notebook 01, section 2.6).", align=["l", "r", "r", "r", "r"])
    H2("4.7 Cleaning log")
    P(f"Every issue, what was checked, how many rows it touched and what was done. The notebook saves it as `reports/cleaning_log.csv` "
      f"({len(LOG)} rows). The raw sample has {n('nb01', '199,580')} rows. After removing {n('nb01', '713')} rows (702 without `closed_date` and 11 with a "
      f"negative duration), the cleaned file has **{n('nb01', '198,867')} rows and {n('nb01', '15')} columns**.")
    rows = [[r["issue"], r["check"], f"{int(r['rows_affected']):,}", r["action"]] for _, r in LOG.iterrows()]
    TABLE(["Issue", "Check", "Rows", "Action"], rows, [2.4, 6.0, 1.5, 6.0], "Cleaning log (`reports/cleaning_log.csv`).", font=7.5,
          align=["l", "l", "r", "l"])
    H2("4.8 Exploratory charts")
    P("These charts describe the cleaned **training period** only, so nothing noticed here comes from validation or test rows.")
    FIG("03_requests_by_hour_weekday.png", "Requests by hour of the day and by weekday (training period).", 15.0)
    P(f"Requests follow the working day: fewest around 4:00 ({n('nb01', '1,652')} requests), most at 10:00 ({n('nb01', '8,377')}). Weekdays are busier than "
      f"weekends, but requests filed at the weekend are resolved within 7 days more often ({n('nb01', '87.9%')} against {n('nb01', '81.0%')}), because weekend requests "
      "are more often noise and parking complaints.")
    FIG("03_share_by_borough_channel.png", "Share resolved within 7 days by borough and by channel (training period).", 15.0)
    P(f"The boroughs are close together, from {n('nb01', '79.8%')} (Manhattan) to {n('nb01', '85.7%')} (Queens). The channel matters more: "
      f"{n('nb01', '95.3%')} for mobile, {n('nb01', '84.9%')} online, {n('nb01', '75.7%')} by phone and {n('nb01', '61.2%')} when the channel is unknown. "
      "Complaint type is the strongest signal: police-handled types (illegal parking, noise) are resolved within 7 days almost every time, while housing "
      f"types such as unsanitary condition ({n('nb01', '39.2%')}) and plumbing ({n('nb01', '46.3%')}) are much slower.")
    H2("4.9 The preprocessing pipeline")
    STEP("One scikit-learn `Pipeline` (`build_pipeline` in `src/features.py`) does everything between a raw request and the model. "
         "Step 1, `clean` (`RequestCleaner`), takes the raw fields exactly as the API will receive them, applies the cleaning rules of this section, "
         "replaces rare or unseen categories with `Other`, adds the four time features and returns **only** the feature columns. "
         "Step 2, `encode` (`ColumnTransformer`), turns them into numbers. Step 3, `model`, is added in Phase 5.",
         "A new request is then treated exactly like the training rows, and because step 1 returns only the 12 feature columns, an excluded column can never reach the model.",
         "Everything the pipeline learns (the list of common categories, the one-hot columns, the median coordinates, the scaling values) is fitted on "
         "the **training split only**. Validation, test and API requests are only transformed.",
         "jupyter nbconvert --to notebook --execute --inplace notebooks/02_feature_selection.ipynb")
    TABLE(["Columns", "Encoding", "Why"],
          [["Text columns (6): `complaint_type`, `descriptor`, `location_type`, `borough`, `incident_zip`, `open_data_channel_type`",
            "`OneHotEncoder(handle_unknown=\"ignore\")`: one 0/1 column per category, stored sparsely",
            "Models need numbers. A category never seen in training gets all zeros instead of an error."],
           ["`latitude`, `longitude`",
            "Fill missing values with the training median, add a 0/1 \"was missing\" column, then `StandardScaler`",
            "There is no honest value for a missing location. Scaling puts both on the same range."],
           ["`hour`, `weekday`, `month`, `is_weekend`", "`StandardScaler`", "Puts the numbers on a comparable range."]],
          [5.0, 5.9, 5.0], "Encoding and scaling in step 2 of the pipeline.", font=8.5)
    P(f"With the 12 final features the output has {n('nb02', '563')} columns ({n('nb02', '555')} one-hot columns from the text fields, 4 for the coordinates and 4 for "
      "the time features) and is stored as a sparse matrix.")
    H3("Training-serving check")
    P("**The risk.** The model is trained on the cleaned file, but the API receives raw fields. If the two paths clean data differently, the model "
      "quietly gets worse (\"training-serving skew\"). Notebook 02, section 5 runs three checks:")
    BUL([f"**Check 1.** Take {n('nb02', '5,000')} random training rows. Fetch the same rows from the **raw** file (all 44 untouched text columns) and run only the "
         "pipeline's `clean` step. The result must equal `clean.parquet` for every column. **Result: PASS** (all 15 columns equal for all 5,000 rows).",
         "**Check 2.** Run the `clean` step on rows that are already clean. Nothing may change. **Result: PASS.**",
         "**Check 3.** Five awkward requests (odd case and spaces, an unseen complaint type, missing or invalid location, only the four required fields). "
         "**Result: PASS.** Nothing crashed and the output had no missing or infinite numbers."])


def section_5():
    cmp_ = CMP.set_index("model")
    models = ["Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees", "XGBoost", "Bernoulli Naive Bayes"]
    H2("5.1 The six candidates and the two baselines")
    P("A model is useful only if it beats a simple rule. Two **baselines** are used. Six **candidate models** are compared with them. "
      "All use the same cleaning step, the same 12 features and the same split. The classifier is added at the end of the pipeline.")
    TABLE(["Model", "Idea", "How it handles the 17% minority class (class 0)"],
          [["Baseline: majority class", "Always answers \"resolved within 7 days\".", "None. Its recall for class 0 is 0."],
           ["Baseline: complaint type only", "For each complaint type, answers with the majority class seen in training.", "Its ranking score is the type's share resolved within 7 days."],
           ["Logistic Regression", "A weighted sum of the inputs, turned into a probability.", "`class_weight=\"balanced\"`"],
           ["Decision Tree", "One tree of yes/no questions.", "`class_weight=\"balanced\"`"],
           ["Random Forest", "Many trees on random samples, averaged.", "`class_weight=\"balanced_subsample\"`"],
           ["Extra Trees", "Like a forest, but each split point is chosen at random.", "`class_weight=\"balanced_subsample\"` (same weights as \"balanced\" here)"],
           ["XGBoost (`XGBClassifier`)", "Trees built one after another, each correcting the last.", "`scale_pos_weight` (below 1 makes the common class 1 count less)"],
           ["Bernoulli Naive Bayes", "Treats every input as a yes/no fact and assumes the facts are independent.", "`class_prior`: equal shares or the real shares (chosen by cross-validation). It has its own all-0/1 encoder."]],
          [3.8, 6.0, 6.1], "Baselines and candidate models.", font=8.5)
    P("`HistGradientBoostingClassifier` and KNN were not used: the pipeline output is sparse, which histogram boosting cannot take. "
      "Random Forest and Extra Trees are kept to 150 and 100 trees with a depth limit, so the saved file stays small enough for a small EC2 server.")
    H2("5.2 Split and metrics")
    P(f"Models are tuned with cross-validation on the **training split** ({n('nb03', '139,206')} rows, class 0 share {n('nb03', '17.2%')}) and compared on the "
      f"**validation split** ({n('nb03', '29,830')} rows, class 0 share {n('nb03', '16.8%')}). The test split is not opened in this phase.")
    P("**Why macro F1 and class 0 recall matter more than accuracy.** About 83% of requests are class 1. A model that always answers \"within 7 days\" "
      f"gets accuracy {cmp_.loc['Baseline: majority class', 'accuracy']:.3f}, but macro F1 only {cmp_.loc['Baseline: majority class', 'macro_f1']:.3f}. "
      "**Class 0 (not resolved within 7 days) is the class staff want to flag.** The metrics are:")
    BUL(["**Recall of class 0:** of the requests that really took longer than 7 days, what share did the model flag?",
         "**Precision of class 0:** of the requests the model flagged, what share really took longer?",
         "**F1:** the balance of precision and recall. **Macro F1** is the average of the F1 of class 0 and class 1, so both classes count equally.",
         "**ROC-AUC:** how well the model ranks slow requests above fast ones, whatever cut-off is used (0.5 = guessing, 1.0 = perfect).",
         "**Confusion matrix:** counts of requests by what really happened and what the model predicted."])
    H2("5.3 Time-ordered cross-validation")
    STEP("`TimeSeriesSplit(n_splits=5)` cuts the training split into 6 blocks by date. Fold 1 trains on block 1 and validates on block 2, fold 2 trains on blocks 1-2 "
         "and validates on block 3, and so on.",
         "A random split would let the model train on December requests and be scored on November ones. It could learn things about that period that it can never "
         "know in real use. In time order, **every fold trains on the past and validates on the next block**, which is what will happen in real use.",
         "The notebook prints the date range of every fold. Two things to remember: early folds train on little data, so their scores are lower and vary more; the first block is never used for validation.",
         "jupyter nbconvert --to notebook --execute --inplace notebooks/03_model_comparison.ipynb")
    t3 = T03[3][0]
    rows = [[str(f), f"{int(t3.loc[f, 'train_rows']):,}", f"{t3.loc[f, 'train_from']} to {t3.loc[f, 'train_to']}",
             f"{int(t3.loc[f, 'validate_rows']):,}", f"{t3.loc[f, 'validate_from']} to {t3.loc[f, 'validate_to']}",
             str(t3.loc[f, "validate_class0_share"])] for f in t3.index]
    TABLE(["Fold", "Train rows", "Train period", "Validate rows", "Validate period", "Class 0 share"], rows, [1.2, 2.1, 4.4, 2.3, 4.4, 1.5],
          "The five time-ordered folds (notebook 03, section 2).", font=8, align=["c", "r", "l", "r", "l", "r"])
    H2("5.4 Light tuning")
    P("Each model has a few settings that change how it behaves. A small number of settings was tried, each scored with the 5 time-ordered folds **on the training split only**. "
      "The setting with the highest mean macro F1 is chosen (ROC-AUC breaks ties). Phase 5 first covered Logistic Regression, Random Forest and XGBoost "
      "(6 to 8 settings each). Phase 5b added Decision Tree, Extra Trees and Bernoulli Naive Bayes (4 settings each), starting from a sensible first setting. "
      "The validation split was not used for tuning.")
    P(f"For XGBoost, two `scale_pos_weight` values were tried: full balance ({n('nb03', '0.2076')}, rows of class 0 divided by rows of class 1) and a milder half-way value "
      f"({n('nb03', '0.4556')}, its square root). The milder one won. For Bernoulli Naive Bayes, the real class shares (`class_prior=None`) beat equal shares in cross-validation.")
    rows = []
    for model in models:
        part = TUNE[TUNE["model"] == model].sort_values("cv_macro_f1_mean", ascending=False)
        for i, (_, r) in enumerate(part.iterrows()):
            settings = ", ".join(f"{k}={v}" for k, v in json.loads(r["settings"]).items())
            rows.append([model if i == 0 else "", settings + ("  (best)" if i == 0 else ""),
                         f"{r['cv_macro_f1_mean']:.4f} ± {r['cv_macro_f1_std']:.4f}",
                         f"{r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f}"])
    TABLE(["Model", "Settings tried (best first)", "CV macro F1 (mean ± std)", "CV ROC-AUC (mean ± std)"], rows, [3.2, 6.3, 3.2, 3.2],
          f"All {len(TUNE)} settings tried, scored by time-ordered cross-validation on the training split (`reports/tuning_results.csv`).",
          font=7.5, align=["l", "l", "r", "r"])
    best_rows = [[m, ", ".join(f"{k}={v}" for k, v in BEST[m].items())] for m in models]
    TABLE(["Model", "Best setting (`reports/best_settings.json`)"], best_rows, [4.5, 11.4], "Best setting of each candidate.", font=8.5)
    H2("5.5 Comparison on the validation split")
    P("Each tuned model was fitted once on the **full training split** and scored on the **validation split**, which the tuning never saw. The two baselines were scored the same way.")
    order = ["Baseline: majority class", "Baseline: complaint type only"] + models
    rows = []
    for m in order:
        r = cmp_.loc[m]
        rows.append([m, f"{r['accuracy']:.3f}", f"{r['precision_0']:.3f}", f"{r['recall_0']:.3f}", f"{r['f1_0']:.3f}", f"{r['macro_f1']:.3f}", f"{r['roc_auc']:.3f}"])
    best_name = max(models, key=lambda m: cmp_.loc[m, "macro_f1"])
    TABLE(["Model", "Accuracy", "Precision (class 0)", "Recall (class 0)", "F1 (class 0)", "Macro F1", "ROC-AUC"], rows, [5.3, 1.7, 1.9, 1.8, 1.7, 1.8, 1.7],
          "Validation results for every model (29,830 requests; `reports/model_comparison.csv`).", font=8.5,
          align=["l", "r", "r", "r", "r", "r", "r"], bold_rows=[order.index(best_name)])
    rows = []
    for m in order:
        r = cmp_.loc[m]
        rows.append([m, f"{r['cv_macro_f1_mean']:.4f} ± {r['cv_macro_f1_std']:.4f}", f"{r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f}",
                     f"{r['train_seconds']:.1f}", f"{r['file_mb']:.2f}", f"{r['predict_one_ms']:.1f}"])
    TABLE(["Model", "CV macro F1", "CV ROC-AUC", "Train (s)", "File (MB)", "One prediction (ms)"], rows, [4.6, 2.9, 2.9, 1.6, 1.7, 2.2],
          "Cross-validation on the training split, and the cost of each model on the development laptop.", font=8.5,
          align=["l", "r", "r", "r", "r", "r"])
    x = cmp_.loc["XGBoost"]
    b = cmp_.loc["Baseline: complaint type only"]
    P("**Reading the results.**")
    BUL([f"**The complaint-type baseline is strong.** The complaint type alone gives macro F1 {b['macro_f1']:.3f} and ROC-AUC {b['roc_auc']:.3f}. This confirms sections 2 to 4: most of the signal is in what the request is about.",
         f"**XGBoost is the best single model** on macro F1 ({x['macro_f1']:.3f}), ROC-AUC ({x['roc_auc']:.3f}) and accuracy ({x['accuracy']:.3f}). It adds "
         f"{x['macro_f1'] - b['macro_f1']:.3f} macro F1 over the baseline and flags {pct(x['recall_0'], 0)} of slow requests instead of {pct(b['recall_0'], 0)}.",
         f"**Logistic Regression** ({cmp_.loc['Logistic Regression', 'macro_f1']:.3f}), **Decision Tree** ({cmp_.loc['Decision Tree', 'macro_f1']:.3f}) and **Bernoulli Naive Bayes** "
         f"({cmp_.loc['Bernoulli Naive Bayes', 'macro_f1']:.3f}) form a second group. The Decision Tree ranks requests worst (ROC-AUC {cmp_.loc['Decision Tree', 'roc_auc']:.3f}, below the baseline), because one tree gives only a coarse set of probabilities.",
         f"**Random Forest** ({cmp_.loc['Random Forest', 'macro_f1']:.3f}) and **Extra Trees** ({cmp_.loc['Extra Trees', 'macro_f1']:.3f}) flag the most slow requests "
         f"({pct(cmp_.loc['Random Forest', 'recall_0'], 0)} and {pct(cmp_.loc['Extra Trees', 'recall_0'], 0)}) but with the most false alarms: only {pct(cmp_.loc['Random Forest', 'precision_0'], 0)} of their flags are right. "
         "Their macro F1 is slightly below the baseline, although they rank requests better (ROC-AUC).",
         "**The class weights explain the pattern.** Weighted models are told to care more about class 0, so they flag more slow requests and pay with more false alarms. XGBoost did best with the milder weight, so it has the fewest false alarms of the weighted models.",
         "**Cross-validation and validation mostly agree.** XGBoost is first in both. The one real disagreement is the Decision Tree: last in cross-validation, third on the validation split.",
         f"**Cost is not a constraint.** Every model trains in under a minute. The largest saved file is {cmp_['file_mb'].max():.0f} MB (Random Forest), and one prediction takes well under 0.1 seconds on the development laptop."])
    P(f"**Best single model: XGBoost**, by validation macro F1 ({x['macro_f1']:.3f}). It is also first on ROC-AUC and in cross-validation.")
    H2("5.6 Charts")
    FIG("05_roc_curves.png", "ROC curves of all models on the validation split.", 11.0, 12.0)
    FIG("05_roc_curves_zoom.png", "The same ROC curves, zoomed in on the top-left corner.", 12.0)
    P("Each confusion matrix counts validation requests by what really happened (rows) and what the model predicted (columns). The top-left cell is the **recall of class 0**.")
    names = [("Logistic Regression", "logistic_regression"), ("Decision Tree", "decision_tree"), ("Random Forest", "random_forest"),
             ("Extra Trees", "extra_trees"), ("XGBoost", "xgboost"), ("Bernoulli Naive Bayes", "bernoulli_naive_bayes")]
    FIG_GRID([(f"05_confusion_{slug}.png", label) for label, slug in names], per_row=3, width_cm=5.0,
             caption="Confusion matrices of the six candidates on the validation split.")
    P("**Feature importance of the best tree model (XGBoost).** Importance is added back up from the one-hot columns to the 12 original features. "
      "It comes from the training data and tends to favour features with many categories, so read it as \"what the model used\", not as proof of what matters.")
    FIG("05_feature_importance_best_tree.png", "Importance of each of the 12 features in XGBoost.", 13.0)


def section_6():
    cmp_ = CMP.set_index("model")
    models = ["Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees", "XGBoost", "Bernoulli Naive Bayes"]
    H2("6.1 The selection rule")
    P("The stacking classifier of section 7 combines three models. A stack helps only if its models make **different** mistakes, so the choice is not simply the top three. "
      "The rule was written **before** the three Phase 5b candidates were run, so that the choice is honest. The test split plays no part in it.")
    BUL(["**Rank** the six candidates by their mean cross-validated macro F1 on the training split. Ties are broken by validation macro F1, then validation ROC-AUC.",
         f"**Take the top 3, with one exception.** If two chosen models have a correlation **above {n('nb03', '0.97')}** between their predicted probabilities on the validation split, the lower-ranked one is dropped and the next-ranked candidate takes its place. This repeats until no chosen pair is above 0.97.",
         "If the candidates run out before three are chosen, the remaining places go to the highest-ranked candidates left, and this is stated."])
    P("**Why use the cross-validation spread.** Cross-validation gives each model five scores, not one. The standard deviation across folds shows how much a score "
      "moves by chance. A difference smaller than that spread should not decide the choice.")
    ranked = cmp_.loc[models].sort_values(["cv_macro_f1_mean", "macro_f1", "roc_auc"], ascending=False)
    H2("6.2 Ranking")
    rows = [[str(i + 1), name, f"{r['cv_macro_f1_mean']:.4f}", f"{r['cv_macro_f1_std']:.4f}", f"{r['macro_f1']:.4f}", f"{r['roc_auc']:.4f}"]
            for i, (name, r) in enumerate(ranked.iterrows())]
    TABLE(["Rank", "Model", "CV macro F1 (mean)", "CV macro F1 (std)", "Validation macro F1", "Validation ROC-AUC"], rows, [1.3, 4.8, 2.5, 2.4, 2.5, 2.4],
          "Ranking of the six candidates (rule step 1; `reports/model_comparison.csv`).", font=8.5, align=["c", "l", "r", "r", "r", "r"])
    top = ranked["cv_macro_f1_mean"]
    gap_first = top.iloc[0] - top.iloc[1]
    spread_2_5 = top.iloc[1] - top.iloc[4]
    P(f"**Reading the spread.** XGBoost leads the next model by {gap_first:.4f}, which is more than its own standard deviation ({ranked['cv_macro_f1_std'].iloc[0]:.4f}), "
      f"so its first place is not luck. The models ranked 2 to 5 are only {spread_2_5:.4f} apart, about one standard deviation, so their order is much less certain. "
      "That is why variety, measured by correlation, is allowed to decide among them.")
    H2("6.3 Correlation between the models")
    P("The correlation is between the predicted probabilities of two models on the validation split (1.00 = they say the same thing). The disagreement is the share of "
      "requests where two models give a different yes/no answer.")
    t1, t2 = T03[22][1], T03[22][2]
    order = list(t1.index)
    rows = [[name] + [f"{float(t1.loc[name, c]):.3f}" for c in order] for name in order]
    short_names = {"Logistic Regression": "LogReg", "Bernoulli Naive Bayes": "Naive Bayes", "Random Forest": "Rand. Forest",
                   "Extra Trees": "Extra Trees", "Decision Tree": "Dec. Tree", "XGBoost": "XGBoost"}
    TABLE(["Model"] + [short_names[c] for c in order], rows, [3.9] + [2.0] * 6,
          "Correlation of predicted probabilities on the validation split (notebook 03, section 8).", font=8.5, align=["l"] + ["r"] * 6)
    rows = [[name] + [str(t2.loc[name, c]) for c in order] for name in order]
    TABLE(["Model"] + [short_names[c] for c in order], rows, [3.9] + [2.0] * 6,
          "Share of validation requests where two models give a different answer.", font=8.5, align=["l"] + ["r"] * 6)
    H2("6.4 Result of the rule")
    rank_names = list(ranked.index)
    P("**How the rule produced the selection.**")
    BUL([f"By cross-validated macro F1 the top three were XGBoost ({top.iloc[0]:.3f}), Logistic Regression ({top.iloc[1]:.3f}) and Bernoulli Naive Bayes ({top.iloc[2]:.3f}).",
         f"XGBoost and Logistic Regression have a correlation of **{n('nb03', '0.975')}**, above the 0.97 limit. They are nearly identical in what they predict, so Logistic Regression (the lower-ranked of the pair) was **replaced**.",
         f"Bernoulli Naive Bayes was accepted: its correlation with XGBoost is {n('nb03', '0.951')}.",
         f"The next-ranked model, **Random Forest**, was accepted: {n('nb03', '0.951')} with XGBoost and {n('nb03', '0.925')} with Naive Bayes."])
    P("**Why Logistic Regression was replaced by Random Forest.** Logistic Regression is the second-best model, but its predictions are almost the same as XGBoost's. "
      "Putting both in a stack adds little. Random Forest is weaker alone but behaves differently, so it can correct XGBoost on some requests. "
      "The rule trades individual strength for variety.")
    selected = [m for m in models if bool(cmp_.loc[m, "selected_for_stack"])]
    assert set(selected) == {"XGBoost", "Bernoulli Naive Bayes", "Random Forest"}, selected
    kinds = {"XGBoost": "Boosted trees", "Bernoulli Naive Bayes": "A simple probability model", "Random Forest": "Averaged trees"}
    rows = []
    for m in ["XGBoost", "Bernoulli Naive Bayes", "Random Forest"]:
        r = cmp_.loc[m]
        rows.append([m, kinds[m], ", ".join(f"{k}={v}" for k, v in BEST[m].items()), f"{r['macro_f1']:.3f}", f"{r['roc_auc']:.3f}", f"{r['recall_0']:.3f}"])
    TABLE(["Selected model", "Kind", "Tuned setting", "Val. macro F1", "Val. ROC-AUC", "Val. recall (class 0)"], rows, [3.2, 3.0, 4.6, 1.7, 1.7, 1.7],
          "The three models selected for the stack (`selected_for_stack` in `reports/model_comparison.csv`).", font=8.5,
          align=["l", "l", "l", "r", "r", "r"])
    P("The three selected models are three different kinds of model, and no pair has a correlation above 0.951.")
    FIG("05_model_comparison_selected.png", "Cross-validated macro F1 of all candidates, with the three selected models marked.", 14.0)
    H2("6.5 How much can the three gain together?")
    t25 = T03[25][0]
    sel_row = t25.iloc[0]
    rows = [[str(i), str(t25.loc[i, "all three right"]), str(t25.loc[i, "all three wrong"]), str(t25.loc[i, "they disagree"]),
             str(t25.loc[i, "at least one right"]), f"{float(t25.loc[i, 'simple average: macro F1']):.3f}"] for i in t25.index]
    TABLE(["Trio", "All right", "All wrong", "Disagree", "At least one right", "Simple average macro F1"], rows, [6.2, 1.8, 1.8, 1.8, 2.2, 2.1],
          "Room for improvement for three possible trios (validation split, notebook 03, section 8).", font=8, align=["l", "r", "r", "r", "r", "r"])
    P(f"A stack can only improve on requests where the three models disagree. For the selected trio the three models are all wrong on the same {sel_row['all three wrong']} of requests, which no combination can fix. "
      f"At least one of them is right for {sel_row['at least one right']}, against {pct(cmp_.loc['XGBoost', 'accuracy'])} for XGBoost alone. "
      f"A plain average of the three scores macro F1 {float(sel_row['simple average: macro F1']):.3f}, which is below XGBoost alone ({cmp_.loc['XGBoost', 'macro_f1']:.3f}). "
      "A stack is smarter than a plain average, because it learns how much to trust each model, but a large gain over XGBoost was not expected. Section 7 will report the real result.")
    P("**Note on how this was decided.** The rule above ranks by cross-validated macro F1 and uses the validation scores and the correlation to confirm and to break ties. "
      "The cross-validation spread is reported in the ranking table so that the reader can judge which differences are real.", italic=True)


def section_7():
    sv = STACKVAL.set_index("model")
    t1 = TESTV1.set_index("model")
    ex = EXPS.set_index("experiment")
    v2 = V2VAL.set_index("model")
    fw = FWD[FWD["test_set"] == "forward test (clean)"].set_index("model")
    fv = FINALVAL.set_index("model")
    f3 = lambda x: f"{x:.3f}"
    f4 = lambda x: f"{x:.4f}"

    def metric_rows(frame, names, labels=None):
        return [[(labels or {}).get(m, m), f3(frame.loc[m, "accuracy"]), f3(frame.loc[m, "precision_0"]), f3(frame.loc[m, "recall_0"]),
                 f3(frame.loc[m, "macro_f1"]), f3(frame.loc[m, "roc_auc"])] for m in names]

    metric_heads = ["Model", "Accuracy", "Class 0 precision", "Class 0 recall", "Macro F1", "ROC-AUC"]
    metric_widths = [4.9, 2.0, 2.3, 2.2, 2.2, 2.3]
    metric_align = ["l", "r", "r", "r", "r", "r"]

    # ------------------------------------------------------------------ 7.1
    H2("7.1 The stacking classifier")
    P("A **stacking classifier** lets several different models each give a probability that a request will be resolved within 7 days. "
      "A small final model, the **final estimator**, then learns how much to trust each of them and gives the final answer. "
      "The idea is that models which make different mistakes can correct each other.")
    BUL(["**Base models.** The three models selected in section 6: XGBoost, Bernoulli Naive Bayes and Random Forest, each with the settings tuned in section 5.",
         "**Final estimator.** A Logistic Regression that takes the three probabilities as input.",
         "**One pipeline.** The saved model is a single scikit-learn pipeline: the shared cleaning step turns the raw fields into the 12 features, then each base model applies its own encoder and gives its probability, then the final estimator combines them.",
         "**Out-of-fold predictions.** The final estimator must learn from probabilities that the base models gave for requests they had **not** been trained on, otherwise it would trust the model that memorises best. "
         "The training split is cut into 5 folds and every training request gets one honest prediction from each base model.",
         "**A known compromise.** Everywhere else the folds respect time (`TimeSeriesSplit`), but a stack needs an out-of-fold prediction for every row, which `TimeSeriesSplit` cannot give. "
         "The stack therefore uses 5-fold stratified cross-validation with shuffling. This stays inside the training split, so the validation and test splits remain in the future of every training request.",
         "**Fitted on the training split only.** The model was not refitted on training plus validation, so the saved file matches the reported scores."])
    rows = []
    for m in ["XGBoost", "Bernoulli Naive Bayes", "Random Forest"]:
        rows.append([m, ", ".join(f"{k}={v}" for k, v in BEST[m].items())])
    TABLE(["Base model", "Tuned setting (from section 5)"], rows, [4.0, 11.9], "The three base models of the first stack (`reports/best_settings.json`).", font=8.5, align=["l", "l"])

    # ------------------------------------------------------------------ 7.2
    H2("7.2 The stack compared with its base models (validation split)")
    names = ["Stack", "XGBoost", "Random Forest", "Bernoulli Naive Bayes", "Simple average of the 3", "Baseline: complaint type only"]
    labels = {"Stack": "**Stack (first version)**"}
    TABLE(metric_heads, metric_rows(sv, names, labels), metric_widths,
          "The first stack against each of its base models, the plain average of the three, and the baseline (`reports/stacking_validation.csv`). Class 0 is \"not resolved within 7 days\".",
          font=8.5, align=metric_align)
    gap_f1 = (sv.loc["Stack", "macro_f1"] - sv.loc["XGBoost", "macro_f1"]) * 100
    P(f"**The stack does not beat XGBoost alone.** Its macro F1 is {f4(sv.loc['Stack', 'macro_f1'])} against {f4(sv.loc['XGBoost', 'macro_f1'])} for XGBoost, "
      f"{abs(gap_f1):.2f} points lower (95% range {n('nb04', '-1.00 to -0.45')} points), and the ROC-AUC is the same ({f4(sv.loc['Stack', 'roc_auc'])} against {f4(sv.loc['XGBoost', 'roc_auc'])}). "
      f"It does better than Random Forest, Naive Bayes and the plain average, and than the complaint-type baseline, but not than its best member.")
    P(f"**What the final estimator learned.** Its weights are {n('nb04', '+4.378')} for XGBoost, {n('nb04', '+5.083')} for Random Forest and {n('nb04', '-1.267')} for Naive Bayes "
      f"(intercept {n('nb04', '-4.150')}). The negative weight is not an error: Naive Bayes is correlated with the other two but more extreme, so the stack uses it as a correction. "
      "Because the three probabilities are so alike, the individual weights are not a reliable ranking of the models.")
    FIG("06_validation_confusion_matrices.png", "Confusion matrices of the first stack and of the other models on the validation split.", 15.5)

    # ------------------------------------------------------------------ 7.3
    H2("7.3 The test split, scored once")
    P("The test split was read exactly once, after the validation results were saved and committed.")
    names = ["Stack", "XGBoost", "Random Forest", "Bernoulli Naive Bayes", "Baseline: complaint type only"]
    TABLE(metric_heads, metric_rows(t1, names, {"Stack": "**Stack (first version)**"}), metric_widths,
          "Test split (29,831 requests, scored once; `reports/test_results.csv`).", font=8.5, align=metric_align)
    stack_drop = (t1.loc["Stack", "macro_f1"] - sv.loc["Stack", "macro_f1"]) * 100
    xgb_drop = (t1.loc["XGBoost", "macro_f1"] - sv.loc["XGBoost", "macro_f1"]) * 100
    P(f"**Same order as on validation.** XGBoost ({f4(t1.loc['XGBoost', 'macro_f1'])}) is ahead of the stack ({f4(t1.loc['Stack', 'macro_f1'])}) by "
      f"{(t1.loc['XGBoost', 'macro_f1'] - t1.loc['Stack', 'macro_f1']) * 100:.2f} points. Macro F1 fell from validation to test by {abs(stack_drop):.2f} points for the stack and "
      f"{abs(xgb_drop):.2f} points for XGBoost. All models fell, and the share of slow requests was higher in the test period ({n('nb04', '18.3%')} against 16.8% in validation), "
      "so this is a change in the data rather than a failure of one model.")

    # ------------------------------------------------------------------ 7.4
    H2("7.4 Trying to improve the stack")
    P("Because the test split had been seen, it could no longer be used to choose anything. Three things were done before any change was made.")
    BUL([f"**A fresh forward test set.** Closed requests created from 2026-09-01 to 2026-10-28, after everything else, pulled with the same script and cleaned with the same rules: "
         f"{n('nb05', '12,069')} requests, with {n('nb05', '15.5%')} in class 0. It is cleaner than the old test split because no model had been scored on it. "
         f"It is also somewhat easier than reality: {n('nb05', '9.2%')} of the requests of that period were still open when the data was pulled and are missing from it.",
         f"**A diagnosis.** At the usual cut-off of 0.5 the stack was {n('nb05', '0.73')} points behind XGBoost. When each model gets a cut-off chosen on the training rows, "
         f"the gap is {n('nb05', '0.08')} points, and {n('nb05', '0.03')} points at each model's best cut-off. The two models have the same ceiling: most of the first gap was only the position of the cut-off.",
         "**Seven experiments** with different base models, a gradient boosting final estimator and a tuned cut-off, all fitted on the training split only and judged on the validation split."])
    FIG("07_macro_f1_by_threshold.png", "Macro F1 against the cut-off, on the training rows (out-of-fold) and on the validation rows.", 15.0)
    rows = []
    for key, r in ex.iterrows():
        rows.append([key, r["description"], f"{r['threshold']:.2f}", f4(r["macro_f1"]), f4(r["roc_auc"]), f"{r['file_mb']:.1f}", f"{r['predict_one_ms']:.0f}"])
    rows.append(["-", "**XGBoost alone** (cut-off 0.5, for comparison)", "0.50", f4(v2.loc["XGBoost alone", "macro_f1"]), f4(v2.loc["XGBoost alone", "roc_auc"]), "1.8", "-"])
    TABLE(["Exp.", "Stack tried", "Cut-off", "Macro F1", "ROC-AUC", "MB", "ms"], rows, [1.2, 7.2, 1.6, 1.7, 1.7, 1.2, 1.3],
          "The seven experiments on the validation split (`reports/stacking_experiments.csv`). MB and ms: size of the models and time for one prediction.",
          font=8, align=["c", "l", "r", "r", "r", "r", "r"])
    P(f"**Result.** All seven stacks lie between {f4(ex['macro_f1'].min())} and {f4(ex['macro_f1'].max())} macro F1, a range smaller than the noise of the validation split. "
      f"The winner by the rule set beforehand (highest validation macro F1) was experiment c2, with macro F1 {f4(ex.loc['c2', 'macro_f1'])}. "
      f"The success criterion fixed in advance (at least equal to XGBoost alone on macro F1, with ROC-AUC not lower) was **{n('nb05', 'NOT MET')}**: it is {abs(ex.loc['c2', 'macro_f1'] - v2.loc['XGBoost alone', 'macro_f1']) * 100:.2f} points below.")
    names = ["Stack v2", "XGBoost alone", "Stack v1 (Phase 6)", "Baseline: complaint type only"]
    TABLE(metric_heads, metric_rows(fw, names, {"Stack v2": "**Improved stack**", "Stack v1 (Phase 6)": "First stack"}), metric_widths,
          "The forward test set (12,069 requests, scored once; `reports/test_forward_results.csv`).", font=8.5, align=metric_align)
    P(f"**On the forward test set the improved stack and XGBoost alone are level** (macro F1 {f4(fw.loc['Stack v2', 'macro_f1'])} against {f4(fw.loc['XGBoost alone', 'macro_f1'])}; the difference is far inside the noise). "
      f"The improvement over the first stack is real ({(fw.loc['Stack v2', 'macro_f1'] - fw.loc['Stack v1 (Phase 6)', 'macro_f1']) * 100:.1f} points of macro F1), but the stack still does not beat XGBoost.")

    # ------------------------------------------------------------------ 7.5
    H2("7.5 Decision: XGBoost alone is the final model")
    P("The course instructor's rule is: **if the stacking model is not as good as XGBoost alone, use XGBoost alone, and explain it clearly.** "
      "That is the case here, so the final model is XGBoost alone. The choice follows from the validation split and the rule; the test numbers agree with it but did not decide it.")
    rows = []
    for label, val, fwd in [("First stack", sv.loc["Stack"], fw.loc["Stack v1 (Phase 6)"]), ("Improved stack", v2.loc["Stack v2"], fw.loc["Stack v2"]),
                            ("**XGBoost alone**", v2.loc["XGBoost alone"], fw.loc["XGBoost alone"])]:
        rows.append([label, f4(val["macro_f1"]), f4(val["roc_auc"]), f4(fwd["macro_f1"]), f4(fwd["roc_auc"])])
    TABLE(["Model", "Validation macro F1", "Validation ROC-AUC", "Forward test macro F1", "Forward test ROC-AUC"], rows, [4.7, 2.8, 2.8, 2.8, 2.8],
          "The two stacks and XGBoost alone, on the validation split and on the forward test set.", font=8.5, align=["l", "r", "r", "r", "r"])
    P("**Why the stack did not help.**")
    BUL([f"**The two models say almost the same thing.** Their probabilities are {n('nb06', '0.976')} correlated and they give a different class for only {n('nb06', '630')} of 29,830 validation requests (2.1%). "
         f"On those, the stack is right {n('nb06', '365')} times and XGBoost {n('nb06', '265')} times: a small gain in accuracy that the stack spends on flagging fewer slow requests.",
         "**There was little room to gain.** The models inside the stack are 0.93 to 0.99 correlated, and the best they could do together was to be right for 1.5 points more requests than XGBoost alone.",
         "**Most of the first gap was only the cut-off** (section 7.4).",
         f"**A stack costs more.** It is 11 times larger ({n('nb06', '19.84')} MB against {n('nb06', '1.81')} MB), more than twice as slow for one prediction, holds four models plus a final estimator, "
         "and its weights are hard to read."])
    P("**This is not a failure of the method.** Stacking helps when the base models make different mistakes. Here almost all of the information sits in two columns (the complaint type and the place), "
      "which every model reads in the same way, so there is nothing for a combination to add. The assignment asked for a meta-classifier to be built and compared with its base models. "
      "That was done; the honest result is that it did not beat its best member.")

    # ------------------------------------------------------------------ 7.6
    H2("7.6 The final model: what it is and what it learned")
    P("**What it is.** One XGBoost model, which builds 600 small decision trees one after another, each correcting the mistakes of the ones before. "
      "It uses the settings tuned in section 5 (depth 8, learning rate 0.05, extra attention to the slow requests) and predicts \"resolved within 7 days\" when its probability is above 0.5. "
      "It is fitted on the training split only.")
    rows = [["Validation split (29,830 requests)", f4(fv.loc["XGBoost alone", "macro_f1"]), f4(fv.loc["XGBoost alone", "roc_auc"]), pct(fv.loc["XGBoost alone", "recall_0"]),
             pct(fv.loc["XGBoost alone", "precision_0"]), pct(fv.loc["XGBoost alone", "accuracy"])],
            ["**Forward test set** (12,069 requests, never used to choose anything)", f4(fw.loc["XGBoost alone", "macro_f1"]), f4(fw.loc["XGBoost alone", "roc_auc"]),
             pct(fw.loc["XGBoost alone", "recall_0"]), pct(fw.loc["XGBoost alone", "precision_0"]), pct(fw.loc["XGBoost alone", "accuracy"])],
            ["Original test split (29,831 requests, scored in Phase 6)", f4(t1.loc["XGBoost", "macro_f1"]), f4(t1.loc["XGBoost", "roc_auc"]), pct(t1.loc["XGBoost", "recall_0"]),
             pct(t1.loc["XGBoost", "precision_0"]), pct(t1.loc["XGBoost", "accuracy"])]]
    TABLE(["Data", "Macro F1", "ROC-AUC", "Slow requests caught", "Slow flags that are right", "Accuracy"], rows, [6.2, 1.8, 1.8, 2.1, 2.1, 1.9],
          "Results of the final model. \"Slow requests caught\" is the recall of class 0 and \"slow flags that are right\" its precision.", font=8.5, align=["l", "r", "r", "r", "r", "r"])
    P("**What it learned.**")
    BUL([f"**It recognises what kind of request was reported and which city service handles it.** The complaint type and the place (`location_type`) together carry between 81% and 93% of the importance "
         f"(gain {n('nb06', '82.5%')}, mean contribution {n('nb06', '80.9%')}, permutation {n('nb06', '93.3%')}). Coordinates, borough, ZIP code and the time of filing together carry about 5%.",
         "**Police-handled requests and heating complaints are almost always closed within 7 days; housing repair requests are the slow ones.** For parking, driveways, abandoned vehicles and noise, "
         "the complaint type and the place together push the prediction strongly towards fast. For plumbing, paint, doors and windows and unsanitary conditions, the place `RESIDENTIAL BUILDING` pushes it towards slow.",
         f"**The place is a fingerprint of the agency.** `STREET/SIDEWALK` is {n('nb06', '98.5%')} police requests and {n('nb06', '99.8%')} are resolved within 7 days; `RESIDENTIAL BUILDING` is {n('nb06', '99.3%')} housing requests "
         f"and only {n('nb06', '63.3%')} are. The agency was left out as a feature because a person filing a request cannot give it, but the place brings much of it back.",
         f"**The spelling of the place can flip a prediction.** The same unsanitary-condition request gets a probability of {n('nb06', '0.447')} (slow) when the place is written `RESIDENTIAL BUILDING` and {n('nb06', '0.986')} (fast) when it is written "
         "`Residential Building/House`. The place must be given the way the 311 system records it."])
    FIG("08_importance_three_ways.png", "What the final model relies on, measured three ways (validation split).", 14.0)
    FIG("08_complaint_type_contributions.png", "The complaint types and places that push hardest towards slow and towards fast (log-odds; validation split).", 15.0)
    P("**Why one request gets its prediction.** XGBoost can split each prediction into one contribution per feature (in log-odds: +1 multiplies the odds of \"fast\" by about 2.7). "
      "For the five sample requests, one or two inputs decide the answer and the rest add very little.")
    FIG("08_sample_explanations.png", "The largest contributions for the five sample requests (validation split).", 16.0)
    P("**How to read the probability.**")
    BUL([f"**It is a score, not an exact chance.** In the middle of the range the real share resolved within 7 days is 7 to 17 points higher than the probability (for example {n('nb06', '82.2%')} really fast for requests scored 0.6 to 0.7), "
         "because the model was trained to pay extra attention to the slow requests. Above 0.9 (64% of all requests) the probability and the real share agree.",
         "**The cut-off of 0.5 is a balance.** A higher cut-off catches more slow requests but raises more false alarms. Macro F1 is flat between 0.4 and 0.6."])
    FIG("08_calibration.png", "The model's probability against the real share resolved within 7 days (validation split).", 9.5)
    P("**What it cannot do.**")
    BUL([f"**It rarely tells which request of a given kind will be slow.** Inside a complaint type its ROC-AUC has a median of only {n('nb06', '0.62')}; it almost never flags Street Condition, Snow or Ice or Traffic Signal requests as slow.",
         f"**It needs the place.** Without `location_type` the macro F1 falls from 0.835 to {n('nb06', '0.652')} and the model catches {n('nb06', '27.7%')} of the slow requests instead of 81.9%. The API returns a warning in that case.",
         "**It does not know new complaint types** (it is practically random on them), and the city renames types from time to time (the water and sewer types were renamed in 2026).",
         "**It uses only what is known when the request is filed.** It cannot know an agency's workload, the weather, or what happens afterwards. The sample holds closed requests only."])

    # ------------------------------------------------------------------ 7.7
    H2("7.7 The saved model and the checks")
    P(f"The final model is saved as one pipeline, `models/model.joblib` (version {META['model_version']}), of {METRICS['file_size_mb']:.2f} MB, built with Python {META['versions']['python']}, "
      f"scikit-learn {META['versions']['scikit-learn']} and XGBoost {META['versions']['xgboost']}. It takes the raw fields of a request and returns a probability, so the API needs no other code than the three files in `src/`. "
      "The file is not stored in Git: it is copied to the server with `scp` (section 9). The two stacks are kept (`models/model_stack_v2.joblib` and `models/model_v1.joblib`) and can be rebuilt from the code.")
    BUL([f"**Fresh-process check.** `python -I src/check_saved_model.py` loads the file in a new Python process, checks its SHA-256 ({META['sha256'][:16]}...) and the library versions against `models/model_meta.json`, "
         "and sends the five sample requests of `data/sample_requests.json` as raw dictionaries. The probabilities equal the ones saved by the notebook, and the five requests sent together give the same answers.",
         "**Reproducible.** `python -m src.train` rebuilds the model from the training split and gives the saved validation scores and the saved probabilities exactly (difference 0).",
         "**Description files.** `models/model_meta.json` lists the 9 raw fields the API receives, the label meaning, the cut-off, the split dates and the library versions; `models/metrics.json` holds the scores."])
    P("**Which numbers are clean.** The forward test set (12,069 requests) was never used to choose anything and is the number to quote. The scores of XGBoost on the original test split were obtained "
      "with settings and a cut-off fixed before that split was opened. The validation scores of the improved stack are slightly optimistic because it was the best of seven experiments.", italic=True)


def section_8():
    PENDING(8, ["The AMI (Ubuntu image) chosen and why.", "The instance type and the reason for the choice, based on the model file size and memory use.",
                "The key pair (the name only; never the key file).", "The security group rules: SSH only from the student's IP, one TCP port (8000) for the API.",
                "The running instance in the EC2 console."],
            ["AMI", "instance type", "key pair (name only, no key contents)", "security group rules", "running instance"])


def section_9():
    PENDING(8, ["The files copied to the server with `scp`, with the same folder layout.", "The virtual environment and the pinned requirements installed on the server (same scikit-learn version as in `model_meta.json`).",
                "The first test run of `uvicorn app.main:app --host 0.0.0.0 --port 8000` from the project root.", "The `systemd` service installed and active, so the API restarts.",
                "`GET /health` answered from the student's laptop at the public address."],
            ["files copied to the server", "requirements installed in the virtual environment", "uvicorn running", "systemd service active", "GET /health from the laptop"])


def section_10():
    PENDING(9, ["A valid `POST /predict` request and its response.", "Invalid requests (a missing required field, a wrong value) and the 422 error that names the field.",
                "The same tests run with `pytest` against `API_URL`.", "The Postman collection saved in `docs/`."],
            ["Postman: valid request", "Postman: response", "Postman: invalid request and its error"])


def section_11():
    PENDING(10, ["The link to the notebook in Google Colab, and a note that it runs from top to bottom."], ["Colab link"])
    TODO("Add the Colab link here when the notebook is shared (not stored in any project file).")


def section_12():
    agency_note = n("nb02", "99.0%")
    P("These are ideas for improving the project. None of them was done.")
    H2("12.1 Predict the duration instead of a class")
    P("The project turns the time to close into a yes/no answer. A **regression** model could predict the number of hours or days. This would give a more useful answer "
      f"(\"about 3 days\"). It is harder, because durations are very uneven: the median is {n('nb01', '7.4')} hours but the longest request took {n('nb01', '926.9')} days (section 4.5). "
      "A model for the logarithm of the duration, or for several quantiles, would be a good first try.")
    H2("12.2 Work out the agency from the complaint type")
    P(f"Section 3 showed that `agency` helped the model, but a person filing a request cannot supply it. The complaint type gives the agency {agency_note} of the time. "
      "The server could look up the agency from a table built on the training data. The notebook already tested a guessed agency, and it gave no clear gain, so this is a low priority.")
    H2("12.3 Use more local data")
    P("Only a 1-in-48 sample of closed requests was used. More of the table could be used, and neighbourhood information could be added (for example the history of the same "
      "ZIP code). Requests that are **still open** could also be used. They were left out, which biases the newest months. Methods made for unfinished events (survival analysis) fit this problem.")
    H2("12.4 Monitoring and retraining")
    P(f"City services change. The share resolved within 7 days moved between {n('nb01', '78.9%')} and {n('nb01', '86.0%')} by month in this sample. The API specification says to log every request and "
      "prediction. These logs could be compared with the real outcome once the request is closed. If the scores fall, the model should be retrained on newer data, "
      "with the same time-ordered split.")
    H2("12.5 Explainability")
    P("Feature importance shows what the model used overall. For one request, staff would like to know **why** it was predicted slow. Tools such as SHAP values could give a "
      "short reason for each prediction.")
    H2("12.6 Hosted deployment")
    P("The API runs on one EC2 server. A more robust set-up could use a container, an Elastic IP or a load balancer, HTTPS, automatic restarts, and a managed service "
      "for models, with deployment scripts so the steps can be repeated.")
    H2("12.7 A stack with more diverse base models")
    P(f"The best models were highly correlated (section 6: {n('nb03', '0.975')} between XGBoost and Logistic Regression), so a stack has little to combine. "
      "A stack would gain more if its models used different information, for example a text model on the descriptor, or a model on location history, "
      "next to the tree models.")


SECTIONS = {"1": section_1, "2": section_2, "3": section_3, "4": section_4, "5": section_5, "6": section_6,
            "7": section_7, "8": section_8, "9": section_9, "10": section_10, "11": section_11, "12": section_12}


# ====================================================================== run
def build(pages):
    fill_cover()
    titles = toc_titles()
    assert [t[0] for t in titles] == [str(i) for i in range(1, 13)], titles
    start_body_section()
    for number, title in titles:
        H1(number, title, first=(number == "1"))
        SECTIONS[number]()
    fill_toc_pages(pages)
    doc.save(str(OUTPUT))


def word_pages():
    """Page of every section heading, read from Word (adjusted page numbers: the body starts at 1)."""
    if not WORD_PAGES_SCRIPT.exists() or shutil.which("powershell") is None:
        return {}
    pdf = SCRATCH / "check.pdf"
    out = SCRATCH / "pages.txt"
    result = subprocess.run(["powershell", "-NoProfile", "-File", str(WORD_PAGES_SCRIPT), "-In", str(OUTPUT), "-Pdf", str(pdf), "-Pages", str(out)],
                            capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        print("Word could not be used to read page numbers:", result.stderr[:300])
        return {}
    pages = {}
    for line in out.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0] == "1":
            match = re.match(r"(\d+)\. ", parts[2])
            if match:
                pages[match.group(1)] = int(parts[1])
    return pages


if __name__ == "__main__":
    pages = json.loads(TOC_FILE.read_text()) if TOC_FILE.exists() and "--fresh" not in sys.argv else {}
    build(pages)
    found = word_pages()
    if found and found != pages:
        # second pass: the TOC is on a page before the body, so filling it does not move the headings
        TOC_FILE.write_text(json.dumps(found, indent=1))
        doc = Document(str(TEMPLATE))
        counters.update({"Figure": 0, "Table": 0})
        BULLET_NUM_ID = make_bullets_numbering()
        build(found)
        print("TOC pages:", found)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print("Saved", OUTPUT)
