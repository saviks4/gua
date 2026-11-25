import base64
import hashlib
import io
import mimetypes
import os
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import groq
import olefile
import openpyxl
import pandas as pd
import pdfplumber
import pyodbc
import streamlit as st
from docx import Document
from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

st.set_page_config(
    page_title="Groq Ассистент (RU интерфейс → UK ответы)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODELS = [
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]

VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

SYSTEM_PROMPT_UK = """Ти — практичний технічний виконавець, який самостійно виконує задачі користувача з базами даних Access (.accdb), таблицями Excel, текстовими документами та скріншотами.
КРИТИЧНІ ПРАВИЛА ВІДПОВІДІ:
1. УСІ свої відповіді, пояснення, висновки, таблиці, формули, SQL-запити та описи зв'язків пиши ВИКЛЮЧНО УКРАЇНСЬКОЮ МОВОЮ.
2. Заборонено відмовлятися від виконання задач або писати відмовки. Якщо користувач ставить задачу по базі .accdb чи скріншоту — одразу виконай її максимально повно й конкретно на основі наданої схеми таблиць, зв'язків, реальних даних та зображень.
3. При роботі з .accdb:
   - описуй підтверджені зв'язки та ймовірні зв'язки між таблицями;
   - давай готовий SQL-запит у синтаксисі Microsoft Access (з квадратними дужками `[Таблиця].[Поле]` для назв із пробілами, кирилицею чи спецсимволами, без коментарів `--` усередині запиту) та одразу наводь результат/висновок на основі отриманих даних;
   - якщо дані в контексті обмежені вибіркою рядків, чітко вкажи це українською та підкажи, яку таблицю чи SQL-запит варто перевірити далі.
"""


def get_api_key() -> str:
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    ui_key = st.session_state.get("groq_api_key_input", "").strip()
    return ui_key or env_key


def build_docx_bytes(answer_text: str, title: str = "Відповідь асистента Groq") -> bytes:
    """Формирует .docx документ в памяти из текста ответа ассистента."""
    doc = Document()
    doc.add_heading(title, level=1)

    for raw_line in (answer_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        clean = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        clean = re.sub(r"__(.*?)__", r"\1", clean)

        if clean.startswith("### "):
            doc.add_heading(clean[4:].strip(), level=3)
        elif clean.startswith("## "):
            doc.add_heading(clean[3:].strip(), level=2)
        elif clean.startswith("# "):
            doc.add_heading(clean[2:].strip(), level=1)
        elif re.match(r"^[-*•]\s+", clean):
            bullet_txt = re.sub(r"^[-*•]\s+", "", clean).strip()
            doc.add_paragraph(bullet_txt, style="List Bullet")
        else:
            doc.add_paragraph(clean)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def extract_sql_blocks_from_text(text: str) -> List[str]:
    """Извлекает SQL-блоки из ответа модели."""
    blocks = re.findall(r"```sql\s*(.*?)```", text or "", flags=re.IGNORECASE | re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


def build_result_zip(
    answer_text: str,
    msg_idx: int,
    extra_sql: str = "",
    result_df: Optional[pd.DataFrame] = None,
    db_filename: str = "",
) -> bytes:
    """Упаковывает ответ в .docx, текст, SQL-запросы и таблицу результатов в .zip-архив."""
    mem = io.BytesIO()
    sql_blocks = extract_sql_blocks_from_text(answer_text)
    if extra_sql.strip():
        sql_blocks.append(extra_sql.strip())

    sql_content = "\n\n;\n\n".join(b.rstrip(";") for b in sql_blocks if b.strip())

    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"zvit_uk_{msg_idx + 1}.docx",
            build_docx_bytes(answer_text, title=f"Звіт / Рішення #{msg_idx + 1}"),
        )
        zf.writestr(f"zvit_uk_{msg_idx + 1}.txt", (answer_text or "").encode("utf-8"))
        if sql_content:
            zf.writestr(f"access_queries_{msg_idx + 1}.sql", (sql_content + ";\n").encode("utf-8"))
        if result_df is not None and not result_df.empty:
            zf.writestr(f"query_result_{msg_idx + 1}.csv", result_df.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr(
            "yak_vikoristati.txt",
            (
                (f"База даних: {db_filename}\n" if db_filename else "")
                + "1. Відкрийте файл zvit_uk_*.docx у Word або WPS Office для оформлення звіту.\n"
                + "2. Якщо у файлі є access_queries_*.sql, скопіюйте SQL-запити у Microsoft Access "
                  "(Створити -> SQL-запит або Конструктор запитів -> Режим SQL) та виконайте їх.\n"
                + "3. Примітка: пряме створення готового бінарного файлу .accdb з формами/звітами "
                  "через pyodbc не підтримується; застосуйте SQL у Access або перенесіть дані з CSV/DOCX.\n"
            ).encode("utf-8"),
        )
    mem.seek(0)
    return mem.getvalue()


def render_message_downloads(
    answer_text: str,
    msg_index: int,
    extra_sql: str = "",
    result_df: Optional[pd.DataFrame] = None,
    db_filename: str = "",
) -> None:
    """Рендерит кнопки скачивания ответа в .docx и готового архива .zip."""
    if not answer_text or answer_text.startswith("Помилка під час звернення до Groq API"):
        return
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        docx_data = build_docx_bytes(answer_text, title=f"Звіт / Відповідь #{msg_index + 1}")
        st.download_button(
            label="📥 Скачать ответ в .docx",
            data=docx_data,
            file_name=f"otvet_assistenta_{msg_index + 1}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key=f"download_docx_{msg_index}",
        )
    with col_d2:
        zip_data = build_result_zip(
            answer_text=answer_text,
            msg_idx=msg_index,
            extra_sql=extra_sql,
            result_df=result_df,
            db_filename=db_filename,
        )
        st.download_button(
            label="📦 Скачать пакет (.zip)",
            data=zip_data,
            file_name=f"access_paket_{msg_index + 1}.zip",
            mime="application/zip",
            key=f"download_zip_{msg_index}",
        )


def encode_image_data_url(file_bytes: bytes, filename: str) -> str:
    """Переводит байты скриншота в data URL для передачи в Groq Chat Completions."""
    mime, _ = mimetypes.guess_type(filename)
    if not mime or not mime.startswith("image/"):
        ext = os.path.splitext(filename.lower())[1]
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.') or 'png'}"
    b64 = base64.b64encode(file_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
    """Извлечение текста из PDF с резервным переходом на pdfplumber для таблиц и сложной верстки."""
    pages_text: List[str] = []
    reader = PdfReader(io.BytesIO(file_bytes))
    num_pages = len(reader.pages)

    empty_pages = 0
    for idx, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        if txt:
            pages_text.append(f"[Файл: {filename} | Сторінка {idx}/{num_pages}]\n{txt}")
        else:
            empty_pages += 1

    if num_pages > 0 and empty_pages > num_pages * 0.7:
        plumber_pages: List[str] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                txt = (page.extract_text() or "").strip()
                tables = page.extract_tables() or []
                tbl_lines = []
                for t in tables:
                    for row in t:
                        cleaned = [c.strip().replace("\n", " ") if c else "" for c in row]
                        if any(cleaned):
                            tbl_lines.append(" | ".join(cleaned))
                combined = "\n".join([x for x in [txt, "\n".join(tbl_lines)] if x]).strip()
                if combined:
                    plumber_pages.append(f"[Файл: {filename} | Сторінка {idx}/{num_pages}]\n{combined}")
        if plumber_pages:
            pages_text = plumber_pages

    full_text = "\n\n".join(pages_text)
    meta = {
        "type": "pdf",
        "pages": num_pages,
        "chars": len(full_text),
    }
    return full_text, meta


def _extract_xml_from_zip_bytes(file_bytes: bytes) -> str:
    """Извлекает текст из ZIP/XML-контейнеров (включая современные форматы WPS Office)."""
    texts: List[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            for name in zf.namelist():
                if name.lower().endswith((".xml", ".rels")):
                    raw = zf.read(name)
                    try:
                        root = ET.fromstring(raw)
                        parts = [el.text.strip() for el in root.iter() if el.text and el.text.strip()]
                        if parts:
                            texts.append(" ".join(parts))
                    except ET.ParseError:
                        decoded = raw.decode("utf-8", errors="ignore")
                        cleaned = re.sub(r"<[^>]+>", " ", decoded)
                        cleaned = re.sub(r"\s+", " ", cleaned).strip()
                        if len(cleaned) > 20:
                            texts.append(cleaned)
    except zipfile.BadZipFile:
        return ""
    return "\n".join(texts)


def _extract_ole_readable_strings(file_bytes: bytes) -> str:
    """Извлекает читаемые строки из бинарных OLE-файлов (.doc, .wps, .et, .dps)."""
    chunks: List[str] = []
    try:
        if olefile.isOleFile(io.BytesIO(file_bytes)):
            with olefile.OleFileIO(io.BytesIO(file_bytes)) as ole:
                for entry in ole.listdir(streams=True, storages=False):
                    data = ole.openstream(entry).read()
                    u16 = re.findall(r"(?:[\x20-\x7e\u0400-\u04ff]\x00){4,}", data)
                    for item in u16:
                        s = item.decode("utf-16-le", errors="ignore").strip()
                        if len(s) >= 4:
                            chunks.append(s)
                    ascii_parts = re.findall(
                        r"[\x20-\x7e\u0410-\u044f\u0401\u0451\u0404\u0454\u0406\u0456\u0407\u0457\u0490\u0491]{5,}",
                        data.decode("cp1251", errors="ignore"),
                    )
                    chunks.extend([p.strip() for p in ascii_parts if p.strip()])
    except Exception:
        pass

    seen = set()
    unique_chunks = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique_chunks.append(c)
    return "\n".join(unique_chunks)


@st.cache_data(show_spinner=False)
def extract_office_or_generic_text(file_bytes: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
    """Универсальный парсер для WPS Office (.wps, .et, .dps), MS Office, таблиц и любых текстовых файлов."""
    ext = os.path.splitext(filename.lower())[1]

    if ext == ".docx":
        doc = Document(io.BytesIO(file_bytes))
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_txt = " | ".join(cell.text.strip().replace("\n", " ") for cell in row.cells)
                if row_txt.strip(" |"):
                    parts.append(row_txt)
        text = "\n".join(f"[Файл: {filename}] {line}" for line in parts)
        return text, {"type": "docx", "chars": len(text)}

    if ext in (".xlsx", ".xlsm"):
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True, read_only=True)
        parts = []
        sheet_count = len(wb.sheetnames)
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows_out = []
            for idx, row in enumerate(ws.iter_rows(values_only=True)):
                if idx > 5000:
                    rows_out.append("... (обмежено 5000 рядками на аркуш для продуктивності)")
                    break
                vals = [str(v).strip() if v is not None else "" for v in row]
                if any(vals):
                    rows_out.append(" | ".join(vals))
            if rows_out:
                parts.append(f"[Файл: {filename} | Аркуш: {sheet_name}]\n" + "\n".join(rows_out))
        wb.close()
        text = "\n\n".join(parts)
        return text, {"type": "xlsx", "sheets": sheet_count, "chars": len(text)}

    if ext == ".csv":
        df = pd.read_csv(io.BytesIO(file_bytes), nrows=5000)
        text = f"[Файл: {filename} | Колонки: {', '.join(map(str, df.columns))}]\n" + df.to_csv(index=False)
        return text, {"type": "csv", "rows": len(df), "chars": len(text)}

    if ext in (".wps", ".et", ".dps", ".doc", ".xls", ".ppt", ".rtf", ".odt", ".ods", ".odp"):
        zip_text = _extract_xml_from_zip_bytes(file_bytes)
        if len(zip_text.strip()) > 50:
            text = f"[Файл: {filename} ({ext})]\n{zip_text}"
            return text, {"type": f"wps_office_xml({ext})", "chars": len(text)}

        ole_text = _extract_ole_readable_strings(file_bytes)
        if len(ole_text.strip()) > 30:
            text = f"[Файл: {filename} ({ext})]\n{ole_text}"
            return text, {"type": f"wps_office_ole({ext})", "chars": len(text)}

        fallback_txt = re.sub(r"\s+", " ", file_bytes.decode("utf-8", errors="ignore")).strip()
        text = f"[Файл: {filename} ({ext})]\n{fallback_txt}"
        return text, {"type": f"raw_fallback({ext})", "chars": len(text)}

    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            decoded = file_bytes.decode(enc)
            text = f"[Файл: {filename}]\n{decoded}"
            return text, {"type": f"text({enc})", "chars": len(text)}
        except UnicodeDecodeError:
            continue

    ole_text = _extract_ole_readable_strings(file_bytes)
    text = f"[Файл: {filename} (бінарний файл)]\n{ole_text or 'Не вдалося витягти текстовий вміст.'}"
    return text, {"type": "binary", "chars": len(text)}


def _get_access_connection(db_path: str) -> pyodbc.Connection:
    """Создаёт подключение к .accdb / .mdb через установленный драйвер Microsoft Access."""
    drivers = [d for d in pyodbc.drivers() if "access" in d.lower()]
    if not drivers:
        raise RuntimeError(
            "В системе не найден драйвер 'Microsoft Access Driver (*.mdb, *.accdb)'. "
            "Установите Microsoft Access Database Engine (разрядности, совпадающей с вашим Python)."
        )
    driver_name = drivers[0]
    conn_str = f"DRIVER={{{driver_name}}};DBQ={db_path};"
    return pyodbc.connect(conn_str, autocommit=True)


def _extract_foreign_keys(cursor: pyodbc.Cursor, tables: List[str]) -> List[Dict[str, str]]:
    """Извлекает явные связи (Foreign Keys) из метаданных Access, если они заданы в схеме."""
    fks: List[Dict[str, str]] = []
    table_set = {t.lower() for t in tables}
    for t_name in tables:
        try:
            for row in cursor.foreignKeys(table=t_name):
                pktable = str(getattr(row, "pktable_name", "") or "")
                pkcol = str(getattr(row, "pkcolumn_name", "") or "")
                fktable = str(getattr(row, "fktable_name", "") or t_name)
                fkcol = str(getattr(row, "fkcolumn_name", "") or "")
                if pktable.lower() in table_set and fktable.lower() in table_set and pkcol and fkcol:
                    fks.append(
                        {
                            "from_table": fktable,
                            "from_col": fkcol,
                            "to_table": pktable,
                            "to_col": pkcol,
                            "kind": "Подтверждённая связь (Foreign Key)",
                        }
                    )
        except Exception:
            continue
    return fks


def _infer_candidate_relationships(
    tables: List[str],
    schema_info: Dict[str, Dict[str, Any]],
    explicit_fks: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Находит вероятные связи между таблицами по совпадающим именам/суффиксам ID или кодов."""
    explicit_pairs = {
        (fk["from_table"].lower(), fk["from_col"].lower(), fk["to_table"].lower(), fk["to_col"].lower())
        for fk in explicit_fks
    }

    col_map: Dict[str, List[Tuple[str, str, str]]] = {}
    for t_name in tables:
        for col in schema_info.get(t_name, {}).get("columns", []):
            cname = col.get("column_name", "")
            if not cname or cname == "ERROR":
                continue
            col_map.setdefault(cname.lower(), []).append((t_name, cname, col.get("type_name", "")))

    candidates: List[Dict[str, str]] = []
    seen_pairs = set()

    for low_col, occurrences in col_map.items():
        if len(occurrences) < 2:
            continue
        is_key_like = bool(re.search(r"(^id$|_id$|^код$|_код$|^code$|_code$|номер|№|fk_)", low_col))
        for i in range(len(occurrences)):
            for j in range(i + 1, len(occurrences)):
                t1, c1, tp1 = occurrences[i]
                t2, c2, tp2 = occurrences[j]
                if t1.lower() == t2.lower():
                    continue
                pair_key = tuple(sorted([(t1.lower(), c1.lower()), (t2.lower(), c2.lower())]))
                if pair_key in seen_pairs:
                    continue
                if (t1.lower(), c1.lower(), t2.lower(), c2.lower()) in explicit_pairs or (
                    t2.lower(),
                    c2.lower(),
                    t1.lower(),
                    c1.lower(),
                ) in explicit_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidates.append(
                    {
                        "from_table": t1,
                        "from_col": c1,
                        "to_table": t2,
                        "to_col": c2,
                        "types": f"{tp1} ↔ {tp2}",
                        "kind": "Вероятная связь по ключу/коду" if is_key_like else "Совпадающее имя поля",
                    }
                )

    return candidates


@st.cache_data(show_spinner=False)
def inspect_accdb_database(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Считывает таблицы, структуру колонок, количество строк и связи из .accdb."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".accdb") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        conn = _get_access_connection(tmp_path)
        cursor = conn.cursor()

        tables: List[str] = []
        for row in cursor.tables(tableType="TABLE"):
            t_name = row.table_name
            if t_name and not t_name.startswith("MSys"):
                tables.append(t_name)

        schema_info: Dict[str, Dict[str, Any]] = {}
        for t_name in tables:
            cols: List[Dict[str, str]] = []
            try:
                for col in cursor.columns(table=t_name):
                    cols.append(
                        {
                            "column_name": str(col.column_name),
                            "type_name": str(col.type_name),
                            "column_size": str(col.column_size or ""),
                            "nullable": "YES" if col.nullable else "NO",
                        }
                    )
            except Exception as e:
                cols.append({"column_name": "ERROR", "type_name": str(e), "column_size": "", "nullable": ""})

            row_count = None
            try:
                cnt_cur = conn.cursor()
                cnt_cur.execute(f"SELECT COUNT(*) FROM [{t_name}]")
                res = cnt_cur.fetchone()
                row_count = int(res[0]) if res else 0
            except Exception:
                row_count = None

            schema_info[t_name] = {
                "columns": cols,
                "row_count": row_count,
            }

        explicit_fks = _extract_foreign_keys(cursor, tables)
        candidate_rels = _infer_candidate_relationships(tables, schema_info, explicit_fks)

        conn.close()
        return {
            "ok": True,
            "filename": filename,
            "tables": tables,
            "schema": schema_info,
            "foreign_keys": explicit_fks,
            "candidate_relationships": candidate_rels,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "filename": filename,
            "tables": [],
            "schema": {},
            "foreign_keys": [],
            "candidate_relationships": [],
            "error": str(exc),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@st.cache_data(show_spinner=False)
def load_accdb_table_data(
    file_bytes: bytes,
    filename: str,
    table_name: str,
    selected_columns: Tuple[str, ...],
    max_rows: int = 1000,
) -> pd.DataFrame:
    """Загружает выбранные колонки и строки из конкретной таблицы .accdb с ограничением max_rows."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".accdb") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        conn = _get_access_connection(tmp_path)
        cols_sql = ", ".join(f"[{c}]" for c in selected_columns) if selected_columns else "*"
        query = f"SELECT TOP {int(max_rows)} {cols_sql} FROM [{table_name}]"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def is_safe_select_sql(sql_query: str) -> Tuple[bool, str]:
    """Проверяет, что SQL-запрос является безопасным запросом на чтение (SELECT) без изменения БД."""
    cleaned = (sql_query or "").strip()
    if not cleaned:
        return False, "SQL-запрос пуст."

    no_comments = re.sub(r"--.*?$", "", cleaned, flags=re.MULTILINE)
    no_comments = re.sub(r"/\*.*?\*/", "", no_comments, flags=re.DOTALL).strip()

    if not re.match(r"^(SELECT|WITH)\b", no_comments, flags=re.IGNORECASE):
        return False, "Разрешены только запросы на чтение данных (SELECT)."

    forbidden = re.search(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|INTO\s+\[?\w+|GRANT|REVOKE)\b",
        no_comments,
        flags=re.IGNORECASE,
    )
    if forbidden:
        return False, f"Запрещённая операция в SQL-запросе: {forbidden.group(0).upper()}."

    return True, "OK"


@st.cache_data(show_spinner=False)
def run_accdb_select_query(
    file_bytes: bytes,
    filename: str,
    sql_query: str,
    max_rows: int = 500,
) -> Tuple[bool, pd.DataFrame, str]:
    """Выполняет безопасный SELECT-запрос к .accdb и возвращает DataFrame."""
    ok, msg = is_safe_select_sql(sql_query)
    if not ok:
        return False, pd.DataFrame(), msg

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".accdb") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        conn = _get_access_connection(tmp_path)
        df = pd.read_sql(sql_query, conn)
        conn.close()
        if len(df) > max_rows:
            df = df.head(max_rows)
        return True, df, "OK"
    except Exception as exc:
        return False, pd.DataFrame(), str(exc)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def build_accdb_overview_text(filename: str, db_info: Dict[str, Any]) -> str:
    """Формирует сводку по всей базе .accdb: таблицы, колонки, типы и связи."""
    lines = [f"[База даних Access: {filename} | Таблиць: {len(db_info.get('tables', []))}]"]
    for t_name in db_info.get("tables", []):
        t_meta = db_info["schema"].get(t_name, {})
        cols_desc = ", ".join(
            f"{c['column_name']} ({c['type_name']})"
            for c in t_meta.get("columns", [])
            if c["column_name"] != "ERROR"
        )
        lines.append(
            f"Таблиця [{t_name}] (рядків: {t_meta.get('row_count', 'невідомо')}): {cols_desc}"
        )

    fks = db_info.get("foreign_keys", [])
    if fks:
        lines.append("\nПідтверджені зв'язки (Foreign Keys):")
        for fk in fks:
            lines.append(
                f"  - [{fk['from_table']}].[{fk['from_col']}] -> [{fk['to_table']}].[{fk['to_col']}]"
            )

    cand = db_info.get("candidate_relationships", [])
    if cand:
        lines.append("\nЙмовірні зв'язки за ключами/назвами полів:")
        for c in cand[:30]:
            lines.append(
                f"  - [{c['from_table']}].[{c['from_col']}] <-> [{c['to_table']}].[{c['to_col']}] ({c['kind']})"
            )

    return "\n".join(lines)


def build_table_schema_text(filename: str, table_name: str, table_meta: Dict[str, Any]) -> str:
    """Формирует текстовое описание структуры таблицы Access для контекста ассистента."""
    lines = [
        f"[База даних Access: {filename} | Таблиця: {table_name} | Всього рядків: {table_meta.get('row_count', 'невідомо')}]",
        "Структура таблиці (колонки та типи):",
    ]
    for c in table_meta.get("columns", []):
        lines.append(
            f"  - {c['column_name']} | тип: {c['type_name']} | розмір: {c['column_size']} | NULL: {c['nullable']}"
        )
    return "\n".join(lines)


def dataframe_to_text_blocks(
    df: pd.DataFrame,
    filename: str,
    table_name: str,
    rows_per_block: int = 25,
) -> List[str]:
    """Преобразует строки таблицы в компактные текстовые блоки для поиска релевантных фрагментов."""
    if df.empty:
        return [f"[Файл: {filename} | Таблиця: {table_name}] Таблиця порожня."]

    blocks: List[str] = []
    total_rows = len(df)
    for start in range(0, total_rows, rows_per_block):
        end = min(start + rows_per_block, total_rows)
        sub_df = df.iloc[start:end]
        row_lines = []
        for idx, r in sub_df.iterrows():
            pairs = [f"{col}={val}" for col, val in r.items() if pd.notna(val) and str(val).strip() != ""]
            if pairs:
                row_lines.append(f"Рядок {idx + 1}: " + "; ".join(pairs))
        if row_lines:
            blocks.append(
                f"[Файл: {filename} | Таблиця: {table_name} | Рядки {start + 1}-{end}]\n" + "\n".join(row_lines)
            )
    return blocks


def chunk_text(text: str, source_label: str, chunk_size: int = 1200, overlap: int = 150) -> List[Dict[str, str]]:
    """Разбивает длинный текст на фрагменты с перекрытием."""
    clean_text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not clean_text:
        return []

    chunks: List[Dict[str, str]] = []
    start = 0
    length = len(clean_text)
    step = max(1, chunk_size - overlap)
    part_num = 1

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            split_pos = clean_text.rfind("\n", start + int(chunk_size * 0.6), end)
            if split_pos == -1:
                split_pos = clean_text.rfind(" ", start + int(chunk_size * 0.6), end)
            if split_pos != -1 and split_pos > start:
                end = split_pos

        fragment = clean_text[start:end].strip()
        if fragment:
            chunks.append(
                {
                    "source": source_label,
                    "chunk_id": f"{source_label} #фрагмент {part_num}",
                    "text": fragment,
                }
            )
            part_num += 1

        if end >= length:
            break
        start = max(start + step, end - overlap)

    return chunks


def retrieve_relevant_chunks(
    query: str,
    chunks: List[Dict[str, str]],
    top_k: int = 4,
    max_context_chars: int = 8000,
) -> List[Dict[str, Any]]:
    """Отбирает наиболее релевантные фрагменты текста под запрос пользователя (TF-IDF + совпадение слов)."""
    if not chunks:
        return []

    corpus = [c["text"] for c in chunks]
    scores: List[float] = [0.0] * len(chunks)

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[\w\-]{2,}\b",
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        for i, s in enumerate(sims):
            scores[i] += float(s) * 10.0
    except ValueError:
        pass

    query_tokens = set(re.findall(r"(?u)\b[\w\-]{2,}\b", query.lower()))
    if query_tokens:
        for i, ch in enumerate(chunks):
            txt_lower = ch["text"].lower()
            src_lower = ch["source"].lower()
            hits = sum(1 for tok in query_tokens if tok in txt_lower)
            src_hits = sum(1 for tok in query_tokens if tok in src_lower)
            scores[i] += (hits / max(1, len(query_tokens))) * 3.0 + src_hits * 1.5

    ranked_indices = sorted(range(len(chunks)), key=lambda idx: scores[idx], reverse=True)

    selected: List[Dict[str, Any]] = []
    used_chars = 0

    for idx in ranked_indices:
        if len(selected) >= top_k:
            break
        ch = chunks[idx]
        if scores[idx] == 0.0 and len(selected) >= min(2, top_k):
            continue
        ch_len = len(ch["text"])
        if selected and (used_chars + ch_len > max_context_chars):
            continue
        if not selected and ch_len > max_context_chars:
            ch = {**ch, "text": ch["text"][:max_context_chars] + "\n... [фрагмент скорочено]"}
            ch_len = len(ch["text"])

        selected.append({**ch, "score": round(scores[idx], 3)})
        used_chars += ch_len

    return selected


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "parsed_files" not in st.session_state:
        st.session_state.parsed_files = {}
    if "accdb_selected_tables" not in st.session_state:
        st.session_state.accdb_selected_tables = {}
    if "accdb_selected_columns" not in st.session_state:
        st.session_state.accdb_selected_columns = {}
    if "accdb_last_sql" not in st.session_state:
        st.session_state.accdb_last_sql = ""


def process_uploaded_files(uploaded_files: List[Any]) -> None:
    """Кэширует загруженные файлы (включая .accdb, документы и скриншоты) по SHA-256 хэшу."""
    for uf in uploaded_files:
        file_bytes = uf.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if file_hash in st.session_state.parsed_files:
            continue

        filename = uf.name
        ext = os.path.splitext(filename.lower())[1]

        if ext in IMAGE_EXTS:
            st.session_state.parsed_files[file_hash] = {
                "name": filename,
                "hash": file_hash,
                "kind": "image",
                "bytes": file_bytes,
            }
        elif ext in (".accdb", ".mdb"):
            db_info = inspect_accdb_database(file_bytes, filename)
            default_tables = db_info["tables"][:3] if db_info.get("tables") else []
            st.session_state.accdb_selected_tables[file_hash] = default_tables
            st.session_state.parsed_files[file_hash] = {
                "name": filename,
                "hash": file_hash,
                "kind": "accdb",
                "bytes": file_bytes,
                "db_info": db_info,
            }
        elif ext == ".pdf":
            text, meta = extract_pdf_text(file_bytes, filename)
            st.session_state.parsed_files[file_hash] = {
                "name": filename,
                "hash": file_hash,
                "kind": "text",
                "text": text,
                "meta": meta,
            }
        else:
            text, meta = extract_office_or_generic_text(file_bytes, filename)
            st.session_state.parsed_files[file_hash] = {
                "name": filename,
                "hash": file_hash,
                "kind": "text",
                "text": text,
                "meta": meta,
            }


def build_knowledge_chunks(
    chunk_size: int,
    overlap: int,
    max_rows_per_table: int,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Собирает все чанки из кэшированных файлов и выбранных таблиц .accdb."""
    all_chunks: List[Dict[str, str]] = []
    active_sources: List[str] = []

    for f_hash, item in st.session_state.parsed_files.items():
        fname = item["name"]
        if item["kind"] == "image":
            continue

        if item["kind"] == "text":
            active_sources.append(fname)
            file_chunks = chunk_text(item["text"], source_label=fname, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(file_chunks)

        elif item["kind"] == "accdb":
            db_info = item["db_info"]
            if not db_info.get("ok"):
                continue

            overview_txt = build_accdb_overview_text(fname, db_info)
            all_chunks.append(
                {
                    "source": f"{fname} [Схема та зв'язки]",
                    "chunk_id": f"{fname}:overview",
                    "text": overview_txt,
                }
            )

            chosen_tables = st.session_state.accdb_selected_tables.get(f_hash, [])
            for t_name in chosen_tables:
                t_meta = db_info["schema"].get(t_name, {})
                active_sources.append(f"{fname} [{t_name}]")

                schema_txt = build_table_schema_text(fname, t_name, t_meta)
                all_chunks.append(
                    {
                        "source": f"{fname} [{t_name}]",
                        "chunk_id": f"{fname}:{t_name}:schema",
                        "text": schema_txt,
                    }
                )

                all_cols = [c["column_name"] for c in t_meta.get("columns", []) if c["column_name"] != "ERROR"]
                col_key = f"{f_hash}:{t_name}"
                chosen_cols = tuple(st.session_state.accdb_selected_columns.get(col_key) or all_cols)

                if chosen_cols:
                    try:
                        df = load_accdb_table_data(
                            item["bytes"],
                            fname,
                            t_name,
                            chosen_cols,
                            max_rows=max_rows_per_table,
                        )
                        row_blocks = dataframe_to_text_blocks(df, fname, t_name, rows_per_block=25)
                        for idx, blk in enumerate(row_blocks, start=1):
                            sub_chunks = chunk_text(
                                blk,
                                source_label=f"{fname} [{t_name}]",
                                chunk_size=chunk_size,
                                overlap=overlap,
                            )
                            for sc in sub_chunks:
                                sc["chunk_id"] = f"{fname}:{t_name}:rows_blk{idx}"
                            all_chunks.extend(sub_chunks)
                    except Exception as exc:
                        all_chunks.append(
                            {
                                "source": f"{fname} [{t_name}]",
                                "chunk_id": f"{fname}:{t_name}:error",
                                "text": f"[Помилка читання даних таблиці {t_name}]: {exc}",
                            }
                        )

    return all_chunks, active_sources


def auto_solve_accdb_task(
    api_key: str,
    model_name: str,
    user_task: str,
    f_hash: str,
    item: Dict[str, Any],
    temperature: float,
    max_rows_per_table: int,
) -> Tuple[str, str, pd.DataFrame, str]:
    """
    Автоматически решает задачу пользователя по .accdb:
    1) формирует сводку по схеме и связям БД;
    2) просит модель сгенерировать SQL-запрос под задачу;
    3) выполняет безопасный SELECT-запрос к .accdb;
    4) формирует итоговый разбор и решение на украинском языке.
    """
    client = groq.Groq(api_key=api_key)
    fname = item["name"]
    db_info = item["db_info"]
    overview_txt = build_accdb_overview_text(fname, db_info)

    sql_prompt_messages = [
        {
            "role": "system",
            "content": (
                "Ти — генератор SQL-запитів для Microsoft Access (.accdb). "
                "За схемою бази даних та задачею користувача сформуй один коректний SELECT-запит у синтаксисі Access SQL "
                "(назви таблиць і полів у квадратних дужках `[Таблиця].[Поле]`, без змінних даних і без коментарів `--`). "
                "Поверни запит у блоці ```sql ... ``` і коротке пояснення українською."
            ),
        },
        {
            "role": "user",
            "content": f"Схема та зв'язки БД:\n{overview_txt}\n\nЗадача користувача: {user_task}",
        },
    ]

    sql_gen_resp = client.chat.completions.create(
        model=model_name,
        messages=sql_prompt_messages,
        temperature=0.1,
        max_completion_tokens=1500,
    )
    sql_gen_text = sql_gen_resp.choices[0].message.content or ""
    sql_blocks = extract_sql_blocks_from_text(sql_gen_text)
    generated_sql = sql_blocks[0] if sql_blocks else ""

    query_result_df = pd.DataFrame()
    query_status_uk = "SQL-запит не згенеровано або задача не потребує вибірки."

    if generated_sql:
        ok, df_res, err_msg = run_accdb_select_query(
            item["bytes"],
            fname,
            generated_sql,
            max_rows=min(200, max_rows_per_table),
        )
        if ok:
            query_result_df = df_res
            if df_res.empty:
                query_status_uk = f"Запит виконано успішно, але рядків не знайдено.\nSQL:\n```sql\n{generated_sql}\n```"
            else:
                preview_csv = df_res.head(50).to_csv(index=False)
                query_status_uk = (
                    f"Запит виконано успішно (отримано рядків: {len(df_res)}, показано до 50):\n"
                    f"SQL:\n```sql\n{generated_sql}\n```\n"
                    f"Дані:\n{preview_csv}"
                )
        else:
            query_status_uk = f"Помилка виконання SQL-запиту ({err_msg}):\n```sql\n{generated_sql}\n```"

    final_messages = [
        {"role": "system", "content": SYSTEM_PROMPT_UK},
        {
            "role": "user",
            "content": (
                f"Виконай наступну задачу користувача по базі даних Access `{fname}`:\n"
                f"ЗАДАЧА: {user_task}\n\n"
                f"ОПИС БАЗИ, ТАБЛИЦЬ ТА ЗВ'ЯЗКІВ:\n{overview_txt}\n\n"
                f"РЕЗУЛЬТАТ АВТОМАТИЧНОГО SQL-ЗАПИТУ:\n{query_status_uk}\n\n"
                "Надай готове рішення українською мовою: опиши використані зв'язки між таблицями, "
                "наведи фінальний SQL-запит (якщо доречно) та сформулюй чітку відповідь/висновок за даними."
            ),
        },
    ]

    final_resp = client.chat.completions.create(
        model=model_name,
        messages=final_messages,
        temperature=temperature,
        max_completion_tokens=4096,
    )
    final_answer = final_resp.choices[0].message.content or ""
    return final_answer, generated_sql, query_result_df, query_status_uk


def _pick_vision_model(model_name: str) -> str:
    low = (model_name or "").lower()
    if any(k in low for k in ("vision", "llama-4", "scout", "maverick")):
        return model_name
    return VISION_MODELS[0]


def stream_groq_response(
    api_key: str,
    model_name: str,
    chat_history: List[Dict[str, Any]],
    retrieved_chunks: List[Dict[str, Any]],
    temperature: float,
    attached_images: Optional[List[Dict[str, Any]]] = None,
):
    """Отправляет запрос в Groq с релевантными фрагментами и скриншотами и стримит ответ на украинском языке."""
    client = groq.Groq(api_key=api_key)
    images = attached_images or []

    context_blocks = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"--- ФРАГМЕНТ #{i} | Джерело: {ch['source']} | Релевантність: {ch['score']} ---\n{ch['text']}"
        )

    context_payload = (
        "\n\n".join(context_blocks)
        if context_blocks
        else "Файли не завантажені або релевантних фрагментів не знайдено."
    )

    messages_to_send: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT_UK},
        {
            "role": "system",
            "content": (
                "Нижче наведено найбільш релевантні фрагменти з файлів/баз даних користувача під поточний запит. "
                "Якщо до запиту додано скріншоти — уважно проаналізуй усе, що на них зображено "
                "(текст, повідомлення про помилки, таблиці, інтерфейс, код або SQL) і дай відповідь українською мовою:\n\n"
                + context_payload
            ),
        },
    ]

    for msg in chat_history[:-1]:
        if msg["role"] in ("user", "assistant"):
            messages_to_send.append({"role": msg["role"], "content": msg["content"]})

    last_msg = chat_history[-1]
    last_text = last_msg.get("content", "")

    if images:
        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": last_text or "Проаналізуй прикріплені скріншоти та дай відповідь українською мовою.",
            }
        ]
        for img in images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_data_url(img["bytes"], img["name"])},
                }
            )
        messages_to_send.append({"role": "user", "content": user_content})
        active_model = _pick_vision_model(model_name)
    else:
        messages_to_send.append({"role": "user", "content": last_text})
        active_model = model_name

    stream = client.chat.completions.create(
        model=active_model,
        messages=messages_to_send,
        temperature=temperature,
        max_completion_tokens=4096,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if delta:
            yield delta


def main() -> None:
    init_session_state()

    with st.sidebar:
        st.header("⚙️ Настройки ассистента")

        st.text_input(
            "Groq API Key",
            value=st.session_state.get("groq_api_key_input", ""),
            type="password",
            key="groq_api_key_input",
            help="Можно задать в файле .env (GROQ_API_KEY) или ввести здесь.",
        )

        model_name = st.selectbox(
            "Модель Groq",
            options=DEFAULT_MODELS,
            index=0,
            help="Для скриншотов при необходимости автоматически выбирается vision-модель.",
        )

        temperature = st.slider(
            "Креативность (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Для точного анализа документов и таблиц лучше держать 0.0–0.3.",
        )

        st.divider()
        st.subheader("📂 Загрузка файлов и скриншотов")
        uploaded_files = st.file_uploader(
            "Выберите файлы (.accdb, PDF, WPS Office, таблицы, документы, скриншоты)",
            accept_multiple_files=True,
            type=None,
            key="file_uploader_widget",
        )

        if uploaded_files:
            process_uploaded_files(uploaded_files)

        st.divider()
        with st.expander("🧠 Параметры поиска по большим файлам"):
            chunk_size = st.slider("Размер фрагмента (символов)", 600, 3000, 1200, 200)
            overlap = st.slider("Перекрытие фрагментов (символов)", 50, 600, 150, 50)
            top_k = st.slider("Сколько фрагментов подбирать на запрос", 2, 12, 4, 1)
            max_context_chars = st.slider("Лимит контекста на запрос (символов)", 4000, 30000, 8000, 2000)
            max_rows_per_table = st.slider("Макс. строк из таблицы .accdb", 100, 10000, 1000, 100)

        if st.button("🗑️ Очистить историю чата", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("♻️ Сбросить кэш файлов", use_container_width=True):
            st.session_state.parsed_files = {}
            st.session_state.accdb_selected_tables = {}
            st.session_state.accdb_selected_columns = {}
            st.session_state.accdb_last_sql = ""
            st.cache_data.clear()
            st.rerun()

    st.title("🤖 Ассистент Groq с поддержкой .accdb, скриншотов, PDF и WPS Office")
    st.caption(
        "Интерфейс на русском языке • Файлы и скриншоты кэшируются между сообщениями • "
        "Все ответы ассистента формируются **на украинском языке**."
    )

    accdb_items: List[Tuple[str, Dict[str, Any]]] = []
    cached_images: List[Dict[str, Any]] = []

    if st.session_state.parsed_files:
        with st.expander("📁 Кэшированные файлы, скриншоты и базы данных Access (.accdb)", expanded=True):
            for f_hash, item in st.session_state.parsed_files.items():
                fname = item["name"]
                if item["kind"] == "image":
                    cached_images.append(item)
                    st.markdown(f"🖼️ **Скриншот:** `{fname}`")
                    st.image(item["bytes"], width=300)
                elif item["kind"] == "text":
                    meta = item.get("meta", {})
                    st.markdown(
                        f"**{fname}** — тип: `{meta.get('type', 'file')}`, "
                        f"символов: `{meta.get('chars', 0):,}`"
                        + (f", страниц: `{meta['pages']}`" if "pages" in meta else "")
                    )
                elif item["kind"] == "accdb":
                    accdb_items.append((f_hash, item))
                    db_info = item["db_info"]
                    st.markdown(f"🗄️ **База Access:** `{fname}`")
                    if not db_info.get("ok"):
                        st.error(f"Не удалось прочитать базу .accdb: {db_info.get('error')}")
                        continue

                    tables = db_info.get("tables", [])
                    if not tables:
                        st.warning("В базе данных не найдено пользовательских таблиц.")
                        continue

                    selected_tables = st.multiselect(
                        f"Выберите таблицы для анализа из `{fname}`:",
                        options=tables,
                        default=st.session_state.accdb_selected_tables.get(f_hash, tables[:2]),
                        key=f"ms_tables_{f_hash}",
                    )
                    st.session_state.accdb_selected_tables[f_hash] = selected_tables

                    fks = db_info.get("foreign_keys", [])
                    cand = db_info.get("candidate_relationships", [])
                    with st.expander(f"🔗 Связи таблиц в `{fname}`", expanded=False):
                        if fks:
                            st.markdown("**Подтверждённые связи (Foreign Keys):**")
                            st.dataframe(pd.DataFrame(fks), use_container_width=True, hide_index=True)
                        else:
                            st.info("Явные связи Foreign Key в метаданных не заданы.")

                        if cand:
                            st.markdown("**Вероятные связи по ключам/совпадающим полям:**")
                            st.dataframe(pd.DataFrame(cand), use_container_width=True, hide_index=True)
                        else:
                            st.caption("Совпадающих полей между таблицами не найдено.")

                    for t_name in selected_tables:
                        t_meta = db_info["schema"].get(t_name, {})
                        cols_list = t_meta.get("columns", [])
                        col_names = [c["column_name"] for c in cols_list if c["column_name"] != "ERROR"]

                        st.markdown(f"**Таблица `{t_name}`** (строк: `{t_meta.get('row_count', '—')}`)")
                        tab_struct, tab_preview = st.tabs(["📐 Структура таблицы", "👁️ Предпросмотр данных"])

                        with tab_struct:
                            if cols_list:
                                st.dataframe(pd.DataFrame(cols_list), use_container_width=True, hide_index=True)
                            else:
                                st.info("Нет данных о структуре колонок.")

                        with tab_preview:
                            col_key = f"{f_hash}:{t_name}"
                            chosen_cols = st.multiselect(
                                f"Колонки таблицы `{t_name}` (если пусто — берутся все):",
                                options=col_names,
                                default=st.session_state.accdb_selected_columns.get(col_key, []),
                                key=f"ms_cols_{f_hash}_{t_name}",
                            )
                            st.session_state.accdb_selected_columns[col_key] = chosen_cols
                            active_cols = tuple(chosen_cols) if chosen_cols else tuple(col_names)

                            try:
                                preview_df = load_accdb_table_data(
                                    item["bytes"],
                                    fname,
                                    t_name,
                                    active_cols,
                                    max_rows=min(100, max_rows_per_table),
                                )
                                st.dataframe(preview_df, use_container_width=True, hide_index=True)
                            except Exception as e:
                                st.error(f"Ошибка предпросмотра таблицы `{t_name}`: {e}")

                    with st.expander(f"⚡ Выполнить SELECT-запрос к `{fname}`", expanded=False):
                        default_tbl = selected_tables[0] if selected_tables else tables[0]
                        sql_input = st.text_area(
                            "SQL-запрос (только SELECT):",
                            value=st.session_state.accdb_last_sql or f"SELECT TOP 20 * FROM [{default_tbl}]",
                            height=90,
                            key=f"sql_area_{f_hash}",
                        )
                        if st.button("▶️ Выполнить SQL", key=f"run_sql_btn_{f_hash}"):
                            st.session_state.accdb_last_sql = sql_input
                            ok_q, df_q, msg_q = run_accdb_select_query(
                                item["bytes"], fname, sql_input, max_rows=min(500, max_rows_per_table)
                            )
                            if ok_q:
                                st.success(f"Строк получено: {len(df_q)}")
                                st.dataframe(df_q, use_container_width=True, hide_index=True)
                            else:
                                st.error(f"Ошибка SQL: {msg_q}")

    valid_accdb = [(h, it) for h, it in accdb_items if it.get("db_info", {}).get("ok")]
    if valid_accdb:
        with st.expander("🚀 Автовыполнение задачи по базе Access (.accdb)", expanded=True):
            db_options = {f"{it['name']}": (h, it) for h, it in valid_accdb}
            chosen_db_label = st.selectbox(
                "Выберите базу .accdb для автоматического выполнения задачи:",
                options=list(db_options.keys()),
                key="auto_solve_db_select",
            )
            task_input = st.text_area(
                "Опишите задачу по этой базе данных (ассистент подберёт связи, SQL и выполнит её на украинском языке):",
                placeholder="Например: Соедини таблицы по связям, найди топ-5 записей по значению и объясни связи таблиц",
                height=85,
                key="auto_solve_task_input",
            )
            if st.button("⚡ Автоматически выполнить задачу по .accdb", type="primary", key="auto_solve_btn"):
                api_key = get_api_key()
                if not api_key:
                    st.error("Укажите **GROQ_API_KEY** в файле `.env` или в боковой панели слева.")
                elif not task_input.strip():
                    st.warning("Введите задачу для выполнения.")
                else:
                    sel_hash, sel_item = db_options[chosen_db_label]
                    with st.spinner("Анализирую схему, связи, выполняю SQL-запрос и готовлю решение на украинском..."):
                        ans_uk, gen_sql, res_df, _ = auto_solve_accdb_task(
                            api_key=api_key,
                            model_name=model_name,
                            user_task=task_input.strip(),
                            f_hash=sel_hash,
                            item=sel_item,
                            temperature=temperature,
                            max_rows_per_table=max_rows_per_table,
                        )
                        if gen_sql:
                            st.session_state.accdb_last_sql = gen_sql

                        st.session_state.messages.append(
                            {"role": "user", "content": f"[Задача по {sel_item['name']}]: {task_input.strip()}"}
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": ans_uk,
                                "sql": gen_sql,
                                "query_df": res_df if not res_df.empty else None,
                                "db_filename": sel_item["name"],
                            }
                        )
                        st.rerun()

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg.get("images"):
                for img_b in msg["images"]:
                    st.image(img_b, width=320)
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("🧾 Использованный SQL-запрос к .accdb"):
                    st.code(msg["sql"], language="sql")
            if isinstance(msg.get("query_df"), pd.DataFrame) and not msg["query_df"].empty:
                with st.expander("📊 Данные из результата SQL-запроса"):
                    st.dataframe(msg["query_df"].head(50), use_container_width=True, hide_index=True)
            if msg["role"] == "assistant":
                render_message_downloads(
                    answer_text=msg["content"],
                    msg_index=idx,
                    extra_sql=msg.get("sql", ""),
                    result_df=msg.get("query_df") if isinstance(msg.get("query_df"), pd.DataFrame) else None,
                    db_filename=msg.get("db_filename", ""),
                )
            if msg.get("sources"):
                with st.expander("🔎 Использованные фрагменты для ответа"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['chunk_id']}** (релевантность: `{s['score']}`)")
                        st.caption(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))

    if cached_images:
        st.caption(f"Прикреплено скриншотов: {len(cached_images)} (будут отправлены с запросом)")

    prompt = st.chat_input("Напишите запрос — ассистент ответит на украинском языке...")

    if prompt:
        api_key = get_api_key()
        if not api_key:
            st.error("Укажите **GROQ_API_KEY** в файле `.env` или в боковой панели слева.")
            st.stop()

        img_bytes_list = [img["bytes"] for img in cached_images]
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
                "images": img_bytes_list,
            }
        )
        with st.chat_message("user"):
            for img_b in img_bytes_list:
                st.image(img_b, width=320)
            st.markdown(prompt)

        all_chunks, _ = build_knowledge_chunks(
            chunk_size=chunk_size,
            overlap=overlap,
            max_rows_per_table=max_rows_per_table,
        )
        relevant_chunks = retrieve_relevant_chunks(
            query=prompt,
            chunks=all_chunks,
            top_k=top_k,
            max_context_chars=max_context_chars,
        )

        with st.chat_message("assistant"):
            try:
                response_text = st.write_stream(
                    stream_groq_response(
                        api_key=api_key,
                        model_name=model_name,
                        chat_history=st.session_state.messages,
                        retrieved_chunks=relevant_chunks,
                        temperature=temperature,
                        attached_images=cached_images,
                    )
                )
                if not isinstance(response_text, str):
                    response_text = "".join(response_text or [])

                new_msg_index = len(st.session_state.messages)
                render_message_downloads(
                    answer_text=response_text,
                    msg_index=new_msg_index,
                )

                if relevant_chunks:
                    with st.expander("🔎 Использованные фрагменты для ответа"):
                        for s in relevant_chunks:
                            st.markdown(f"**{s['chunk_id']}** (релевантность: `{s['score']}`)")
                            st.caption(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "sources": relevant_chunks,
                    }
                )
            except Exception as exc:
                err_msg = f"Помилка під час звернення до Groq API: {exc}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})


if __name__ == "__main__":
    main()