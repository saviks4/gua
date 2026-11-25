import ast
import base64
import csv
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="Groq Assistant — Проверка заданий",
    page_icon="✅",
    layout="centered",
)

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_IMAGES = 3
MAX_TOTAL_TEXT_CHARS = 45_000

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

TEXT_EXTS = {
    ".txt",
    ".md",
    ".py",
    ".cs",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".csv",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".sql",
    ".log",
    ".ini",
    ".cfg",
}
IMAGE_EXTS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

SECRET_PATTERNS = [
    r"gsk_[A-Za-z0-9_\-]{10,}",
    r"sk-[A-Za-z0-9_\-]{16,}",
    r"ghp_[A-Za-z0-9]{20,}",
    r"AIza[0-9A-Za-z_\-]{20,}",
]

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "ZeroDivisionError": ZeroDivisionError,
    "AssertionError": AssertionError,
}

BLOCKED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "pathlib",
    "importlib",
    "ctypes",
    "multiprocessing",
    "threading",
    "pickle",
    "builtins",
}


def get_api_key() -> str | None:
    try:
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    env_key = os.getenv("GROQ_API_KEY")
    return env_key.strip() if env_key else None


def audit_code_or_data(ext: str, content: str, label: str) -> list[str]:
    notes: list[str] = []
    lines = content.splitlines()

    if len(lines) > 1500:
        notes.append(f"В «{label}» много строк ({len(lines)}) — проверьте структуру и дублирование.")

    if ext == ".py":
        try:
            tree = ast.parse(content, filename=label)
            has_defs = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in tree.body)
            if content.strip() and not has_defs and len(lines) > 25:
                notes.append(f"В Python «{label}» нет функций или классов — проверьте декомпозицию.")
        except SyntaxError as exc:
            notes.append(
                f"Синтаксическая ошибка в Python «{label}» (строка {exc.lineno}, кол. {exc.offset}): {exc.msg}."
            )
        if re.search(r"except\s*:", content):
            notes.append(f"В «{label}» найден `except:` без типа исключения.")
        if "input(" in content:
            notes.append(f"В «{label}» используется `input()` — для автотестов лучше выносить логику в функции.")

    elif ext == ".json":
        try:
            parsed_json = json.loads(content)
            if isinstance(parsed_json, (dict, list)) and len(parsed_json) == 0:
                notes.append(f"JSON «{label}» пуст.")
        except json.JSONDecodeError as exc:
            notes.append(f"Ошибка JSON в «{label}» (строка {exc.lineno}): {exc.msg}.")

    elif ext == ".csv":
        try:
            dialect = csv.Sniffer().sniff(content[:4096], delimiters=",;\t|")
            rows = list(csv.reader(io.StringIO(content), dialect))
            lengths = {len(r) for r in rows if r}
            if len(lengths) > 1:
                notes.append(f"В CSV «{label}» разное число столбцов ({sorted(lengths)}).")
        except Exception as exc:
            notes.append(f"Не удалось проверить CSV «{label}»: {exc}")

    elif ext == ".html":
        if "<html" not in content.lower() and "<body" not in content.lower():
            notes.append(f"HTML «{label}» выглядит как фрагмент без `<html>`/`<body>`.")
        if "<img" in content.lower() and "alt=" not in content.lower():
            notes.append(f"В HTML «{label}» есть `<img>` без `alt`.")

    elif ext == ".cs":
        if content.count("{") != content.count("}"):
            notes.append(f"В C# «{label}» не совпадает количество фигурных скобок.")

    elif ext == ".sql":
        upper_sql = content.upper()
        if any(k in upper_sql for k in ["DROP TABLE", "TRUNCATE ", "DELETE FROM"]) and "WHERE" not in upper_sql:
            notes.append(f"В SQL «{label}» деструктивная команда без `WHERE`.")

    return notes


def parse_file(uploaded_file: Any) -> dict[str, Any]:
    name = uploaded_file.name
    ext = Path(name).suffix.lower()
    raw = uploaded_file.getvalue()

    result: dict[str, Any] = {
        "name": name,
        "ext": ext,
        "size": len(raw),
        "text": "",
        "image_url": None,
        "error": None,
        "warnings": [],
    }

    if len(raw) == 0:
        result["warnings"].append(f"Файл «{name}» пустой (0 байт).")
        return result

    if len(raw) > MAX_FILE_BYTES:
        result["error"] = f"Файл «{name}» превышает лимит {MAX_FILE_BYTES // (1024 * 1024)} МБ."
        return result

    if ext in TEXT_EXTS:
        decoded = raw.decode("utf-8", errors="replace")
        result["text"] = decoded
        result["warnings"].extend(audit_code_or_data(ext, decoded, name))
        return result

    if ext == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            pages = [f"[Стр. {i + 1}]\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages)]
            text = "\n\n".join(pages).strip()
            if not "".join(p.extract_text() or "" for p in reader.pages).strip():
                result["warnings"].append(f"В PDF «{name}» нет текстового слоя (возможно, скан).")
            result["text"] = text
            return result
        except Exception as exc:
            result["error"] = f"Ошибка чтения PDF «{name}»: {exc}"
            return result

    if ext == ".docx":
        try:
            from docx import Document

            doc = Document(io.BytesIO(raw))
            parts: list[str] = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for idx, table in enumerate(doc.tables, start=1):
                parts.append(f"[Таблица {idx}]")
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))
            text = "\n".join(parts).strip()
            if not text:
                result["warnings"].append(f"Документ «{name}» пуст.")
            result["text"] = text
            return result
        except Exception as exc:
            result["error"] = f"Ошибка чтения DOCX «{name}»: {exc}"
            return result

    if ext in IMAGE_EXTS:
        b64 = base64.b64encode(raw).decode("ascii")
        result["image_url"] = f"data:{IMAGE_EXTS[ext]};base64,{b64}"
        result["text"] = f"[Изображение: {name}, {len(raw)} байт]"
        return result

    result["error"] = f"Формат «{ext or 'без расширения'}» для файла «{name}» не поддерживается."
    return result


def extract_python_candidates(task_text: str, parsed_files: list[dict[str, Any]]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for pf in parsed_files:
        if pf.get("ext") == ".py" and pf.get("text"):
            candidates.append((pf["name"], pf["text"]))

    fenced = re.findall(
        r"```(?:python|py)\s*\n(.*?)```",
        task_text or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    for i, block in enumerate(fenced, start=1):
        candidates.append((f"task_snippet_{i}.py", block))
    return candidates


def has_unsafe_ast(tree: ast.AST) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod in BLOCKED_IMPORTS:
                    return f"Запрещённый импорт модуля `{alias.name}` в автотестах."
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in BLOCKED_IMPORTS:
                return f"Запрещённый импорт из модуля `{node.module}` в автотестах."
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "compile", "__import__", "open"}:
                return f"Запрещённый вызов `{node.func.id}()` в автотестах."
    return None


def generate_and_run_tests(
    api_key: str,
    task_text: str,
    py_candidates: list[tuple[str, str]],
) -> dict[str, Any]:
    """Генерирует assert-тесты под код и запускает их в ограниченном пространстве."""
    if not py_candidates:
        return {"ran": False, "summary": "Python-код не найден — автотесты пропущены.", "results": []}

    source_name, source_code = py_candidates[0]
    try:
        code_tree = ast.parse(source_code, filename=source_name)
    except SyntaxError as exc:
        return {
            "ran": False,
            "summary": f"Автотесты не запущены: синтаксическая ошибка в `{source_name}` (строка {exc.lineno}).",
            "results": [],
        }

    unsafe_reason = has_unsafe_ast(code_tree)
    if unsafe_reason:
        return {
            "ran": False,
            "summary": f"Автотесты остановлены: {unsafe_reason}",
            "results": [],
        }

    client = Groq(api_key=api_key)
    test_prompt = (
        "Напиши 4-6 коротких независимых тестов на чистом Python с использованием `assert` "
        "для проверки функций/классов из кода ниже. Не используй сторонние библиотеки, файлы и сеть.\n"
        "Верни ТОЛЬКО код внутри блока ```python ... ``` без лишнего текста.\n\n"
        f"ЗАДАНИЕ:\n{task_text or '(условие не указано)'}\n\n"
        f"КОД ({source_name}):\n```python\n{source_code[:8000]}\n```"
    )

    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "Ты пишешь точные unit-тесты на assert для Python."},
            {"role": "user", "content": test_prompt},
        ],
    )
    raw_tests = resp.choices[0].message.content or ""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", raw_tests, flags=re.DOTALL | re.IGNORECASE)
    tests_code = (m.group(1) if m else raw_tests).strip()

    try:
        test_tree = ast.parse(tests_code, filename="generated_tests.py")
    except SyntaxError as exc:
        return {
            "ran": False,
            "summary": f"Сгенерированные тесты содержат синтаксическую ошибку: {exc.msg}.",
            "results": [],
        }

    test_unsafe = has_unsafe_ast(test_tree)
    if test_unsafe:
        return {
            "ran": False,
            "summary": f"Сгенерированные тесты заблокированы: {test_unsafe}",
            "results": [],
        }

    env_ns: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(compile(code_tree, source_name, "exec"), env_ns)
    except Exception as exc:
        return {
            "ran": False,
            "summary": f"Ошибка при загрузке кода `{source_name}`: {type(exc).__name__}: {exc}",
            "results": [],
        }

    assert_nodes = [n for n in ast.walk(test_tree) if isinstance(n, ast.Assert)]
    if not assert_nodes:
        # Выполняем весь блок тестов целиком, если assert-узлы сгруппированы иначе
        try:
            exec(compile(test_tree, "generated_tests.py", "exec"), env_ns)
            return {
                "ran": True,
                "summary": f"Тесты для `{source_name}` выполнены успешно.",
                "results": [{"name": "Тестовый блок", "passed": True, "detail": "Выполнен без ошибок"}],
            }
        except Exception as exc:
            return {
                "ran": True,
                "summary": f"Тесты для `{source_name}` завершились с ошибкой.",
                "results": [{"name": "Тестовый блок", "passed": False, "detail": f"{type(exc).__name__}: {exc}"}],
            }

    results: list[dict[str, Any]] = []
    # Сначала выполним вспомогательные определения (функции/переменные) из тестов
    helper_nodes = [n for n in test_tree.body if not isinstance(n, ast.Assert)]
    if helper_nodes:
        helper_mod = ast.Module(body=helper_nodes, type_ignores=[])
        ast.fix_missing_locations(helper_mod)
        try:
            exec(compile(helper_mod, "test_helpers.py", "exec"), env_ns)
        except Exception as exc:
            return {
                "ran": True,
                "summary": f"Ошибка в подготовке тестов: {type(exc).__name__}: {exc}",
                "results": [{"name": "Подготовка тестов", "passed": False, "detail": str(exc)}],
            }

    for idx, stmt in enumerate(assert_nodes, start=1):
        single_mod = ast.Module(body=[stmt], type_ignores=[])
        ast.fix_missing_locations(single_mod)
        expr_hint = ast.unparse(stmt.test) if hasattr(ast, "unparse") else f"Тест #{idx}"
        try:
            exec(compile(single_mod, f"test_{idx}.py", "exec"), env_ns)
            results.append({"name": f"Тест #{idx}: {expr_hint}", "passed": True, "detail": "PASS"})
        except AssertionError:
            results.append({"name": f"Тест #{idx}: {expr_hint}", "passed": False, "detail": "FAIL (AssertionError)"})
        except Exception as exc:
            results.append({"name": f"Тест #{idx}: {expr_hint}", "passed": False, "detail": f"{type(exc).__name__}: {exc}"})

    passed_cnt = sum(1 for r in results if r["passed"])
    summary = f"Автотесты для `{source_name}`: пройдено {passed_cnt} из {len(results)}."
    return {"ran": True, "summary": summary, "results": results}


def detect_profile(task_text: str, files_text: str, images_count: int) -> str:
    combined = f"{task_text}\n{files_text[:4000]}".lower()
    if images_count > 0:
        return "визуальное задание / скриншот / схема"
    if any(k in combined for k in ["def ", "class ", "import ", "select ", "html", "css", "код", "функция", "баг"]):
        return "программирование / код / разметка / запросы"
    if any(k in combined for k in ["формул", "уравнен", "интеграл", "производн", "вероятност", "расчёт"]):
        return "математика / физика / расчёты"
    if any(k in combined for k in ["таблиц", "csv", "столбц", "метрик", "анализ"]):
        return "данные / таблицы / аналитика"
    return "текстовое / учебное / аналитическое задание"


def validate_submission(
    api_key: str | None,
    task_text: str,
    criteria_text: str,
    uploaded_files: list[Any] | None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    clean_task = (task_text or "").strip()
    clean_criteria = (criteria_text or "").strip()

    if not api_key:
        errors.append("Не найден GROQ_API_KEY (проверьте Secrets на Streamlit Cloud или окружение).")
    else:
        checks.append("API-ключ OK")
        if not api_key.startswith("gsk_"):
            warnings.append("Ключ не начинается с 'gsk_' — проверьте корректность GROQ_API_KEY.")

    has_files = bool(uploaded_files)
    if not clean_task and not has_files:
        errors.append("Введите текст задания или прикрепите файлы.")
    else:
        if clean_task:
            checks.append(f"Задание: {len(clean_task)} симв.")
            for idx, snip in enumerate(
                re.findall(r"```(?:python|py)\s*\n(.*?)```", clean_task, flags=re.DOTALL | re.IGNORECASE),
                start=1,
            ):
                warnings.extend(audit_code_or_data(".py", snip, f"фрагмент кода #{idx}"))

    if clean_criteria:
        checks.append(f"Методичка/критерии: {len(clean_criteria)} симв.")
    else:
        warnings.append("Методичка/критерии не заданы — проверка оформления пройдёт по общим стандартам.")

    parsed_files: list[dict[str, Any]] = []
    text_blocks: list[str] = []
    images: list[dict[str, str]] = []
    total_chars = 0
    valid_count = 0

    if uploaded_files:
        for f in uploaded_files:
            pf = parse_file(f)
            parsed_files.append(pf)
            if pf["error"]:
                errors.append(pf["error"])
                continue
            warnings.extend(pf["warnings"])
            valid_count += 1

            if pf["image_url"]:
                if len(images) < MAX_IMAGES:
                    images.append({"name": pf["name"], "url": pf["image_url"]})
                else:
                    warnings.append(f"Картинка «{pf['name']}» пропущена (лимит {MAX_IMAGES}).")

            if pf["text"]:
                rem = MAX_TOTAL_TEXT_CHARS - total_chars
                if rem <= 0:
                    warnings.append("Достигнут лимит объёма текста из файлов — часть текста обрезана.")
                    break
                clipped = pf["text"][:rem]
                total_chars += len(clipped)
                text_blocks.append(f"=== ФАЙЛ: {pf['name']} ({pf['size']} Б) ===\n{clipped}")

        if valid_count > 0:
            checks.append(f"Файлов: {valid_count} (картинок: {len(images)})")

    files_text = "\n\n".join(text_blocks)
    profile = detect_profile(clean_task, files_text, len(images))
    py_candidates = extract_python_candidates(clean_task, parsed_files)
    if py_candidates:
        checks.append(f"Найдено Python-кода для тестов: {len(py_candidates)}")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "profile": profile,
        "task_text": clean_task,
        "criteria_text": clean_criteria,
        "files_text": files_text,
        "images": images,
        "py_candidates": py_candidates,
    }


def sanitize_output(answer: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    text = (answer or "").strip()
    if not text:
        return "", ["Модель вернула пустой ответ."]

    for pat in SECRET_PATTERNS:
        rx = re.compile(pat)
        if rx.search(text):
            text = rx.sub("[СКРЫТЫЙ_СЕКРЕТ]", text)
            notes.append("Обнаружен фрагмент, похожий на ключ/токен — он скрыт в выводе.")

    return text, notes


def run_evaluation(
    api_key: str,
    profile: str,
    task_text: str,
    criteria_text: str,
    files_text: str,
    images: list[dict[str, str]],
    autotest_report: dict[str, Any] | None,
) -> str:
    client = Groq(api_key=api_key)

    system_prompt = (
        "Ты строгий и практичный проверяющий ассистент. Отвечай на языке задания (по умолчанию на русском).\n"
        "Обязательная структура ответа:\n"
        "1. **Вердикт и оценка:** статус (зачтено / есть замечания / требуется доработка) и краткий итог.\n"
        "2. **Ошибки и слабые места:** конкретные недочёты по логике, коду, расчётам или содержанию.\n"
        "3. **Проверка оформления по методичке/критериям:** что выполнено, а что нарушено или отсутствует.\n"
        "4. **Итоги автотестов:** учти результаты запуска тестов (если они выполнялись).\n"
        "5. **Готовое решение / исправленный вариант:** полный рабочий вариант.\n"
        "6. **Уточнения (если нужны):** только критичные вопросы."
    )

    parts = [
        f"ПРОФИЛЬ: {profile}",
        f"ЗАДАНИЕ:\n{task_text or '(текст не введён, см. файлы)'}",
        f"МЕТОДИЧКА / КРИТЕРИИ ОФОРМЛЕНИЯ:\n{criteria_text or '(не заданы — проверь по общим требованиям к аккуратности и полноте)'}",
    ]
    if files_text:
        parts.append(f"ФАЙЛЫ:\n{files_text}")
    if autotest_report and autotest_report.get("summary"):
        lines = [autotest_report["summary"]]
        for r in autotest_report.get("results", []):
            lines.append(f"- {r['name']}: {r['detail']}")
        parts.append("РЕЗУЛЬТАТЫ АВТОТЕСТОВ:\n" + "\n".join(lines))

    payload = "\n\n".join(parts)

    if images:
        user_content: Any = [{"type": "text", "text": payload}]
        for img in images:
            user_content.append({"type": "image_url", "image_url": {"url": img["url"]}})
        model_name = VISION_MODEL
    else:
        user_content = payload
        model_name = TEXT_MODEL

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content or ""


def build_report_markdown(
    title: str,
    val: dict[str, Any],
    autotest_report: dict[str, Any] | None,
    answer: str,
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Отчёт проверки: {title}",
        f"- Дата: {now_str}",
        f"- Профиль: {val.get('profile', '-')}",
        "",
        "## Валидатор",
    ]
    for c in val.get("checks", []):
        lines.append(f"- OK: {c}")
    for w in val.get("warnings", []):
        lines.append(f"- Предупреждение: {w}")
    for e in val.get("errors", []):
        lines.append(f"- Ошибка: {e}")

    if autotest_report and autotest_report.get("summary"):
        lines.extend(["", "## Автопроверка тестов", autotest_report["summary"]])
        for r in autotest_report.get("results", []):
            lines.append(f"- [{'PASS' if r['passed'] else 'FAIL'}] {r['name']} — {r['detail']}")

    lines.extend(["", "## Разбор и решение", answer])
    return "\n".join(lines)


if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Проверка заданий")
st.caption("Задание, критерии/методичка и файлы в одном блоке — валидатор, автотесты, разбор и выгрузка отчёта.")

with st.form("main_check_form", clear_on_submit=False):
    task_input = st.text_area(
        "Задание или решение на проверку",
        height=170,
        placeholder="Вставьте условие задания, код, решение или вопрос...",
    )
    criteria_input = st.text_area(
        "Методичка / требования к оформлению (необязательно)",
        height=90,
        placeholder="Например: наличие титульного/комментариев, названия функций, формат вывода, требования методички...",
    )
    uploaded_files = st.file_uploader(
        "Файлы (код, текст, PDF, DOCX, CSV, JSON, SQL, скриншоты)",
        accept_multiple_files=True,
        type=[
            "txt",
            "md",
            "py",
            "cs",
            "html",
            "css",
            "js",
            "ts",
            "csv",
            "json",
            "xml",
            "yaml",
            "yml",
            "toml",
            "sql",
            "pdf",
            "docx",
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
    )
    run_autotests = st.checkbox("Автопроверка тестов для Python-кода", value=True)
    submitted = st.form_submit_button("Проверить", type="primary", use_container_width=True)

if submitted:
    api_key = get_api_key()
    val = validate_submission(
        api_key=api_key,
        task_text=task_input,
        criteria_text=criteria_input,
        uploaded_files=uploaded_files,
    )

    autotest_report: dict[str, Any] | None = None
    answer = ""
    post_notes: list[str] = []

    if val["ok"] and api_key:
        if run_autotests and val["py_candidates"]:
            with st.spinner("Запускаю автотесты..."):
                autotest_report = generate_and_run_tests(
                    api_key=api_key,
                    task_text=val["task_text"],
                    py_candidates=val["py_candidates"],
                )
        elif run_autotests:
            autotest_report = {"ran": False, "summary": "Python-код не найден — автотесты не запускались.", "results": []}

        with st.spinner("Проверяю задание и оформление..."):
            try:
                raw_answer = run_evaluation(
                    api_key=api_key,
                    profile=val["profile"],
                    task_text=val["task_text"],
                    criteria_text=val["criteria_text"],
                    files_text=val["files_text"],
                    images=val["images"],
                    autotest_report=autotest_report,
                )
                answer, post_notes = sanitize_output(raw_answer)
            except Exception as exc:
                val["errors"].append(f"Ошибка Groq API: {exc}")
                val["ok"] = False

    preview_title = (
        val["task_text"][:60]
        if val["task_text"]
        else (", ".join(f.name for f in (uploaded_files or []))[:60] or "Задание")
    )
    report_md = build_report_markdown(preview_title, val, autotest_report, answer) if answer else ""

    st.session_state.last_report = {
        "title": preview_title,
        "val": val,
        "autotest": autotest_report,
        "notes": post_notes,
        "answer": answer,
        "report_md": report_md,
    }
    if answer:
        st.session_state.history.insert(0, {"title": preview_title, "answer": answer})
        st.session_state.history = st.session_state.history[:10]

active = st.session_state.last_report
if active:
    val = active["val"]
    st.markdown("### Валидатор")
    if val["ok"]:
        st.success("Валидация пройдена: " + " | ".join(val["checks"]))
    else:
        for err in val["errors"]:
            st.error(err)
    for warn in val["warnings"]:
        st.warning(warn)

    auto = active.get("autotest")
    if auto and auto.get("summary"):
        st.markdown("### Автопроверка тестов")
        st.info(auto["summary"])
        for r in auto.get("results", []):
            icon = "✅" if r["passed"] else "❌"
            st.write(f"{icon} **{r['name']}** — `{r['detail']}`")

    for note in active.get("notes", []):
        st.warning(note)

    if active.get("answer"):
        st.markdown("### Результат проверки")
        st.markdown(active["answer"])

        st.markdown("### Выгрузка отчёта")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Скачать отчёт (.md)",
                data=active["report_md"].encode("utf-8"),
                file_name="report_check.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "Скачать отчёт (.txt)",
                data=active["report_md"].encode("utf-8"),
                file_name="report_check.txt",
                mime="text/plain",
                use_container_width=True,
            )

if st.session_state.history:
    with st.expander(f"История проверок ({len(st.session_state.history)})", expanded=False):
        for idx, item in enumerate(st.session_state.history, start=1):
            st.markdown(f"**{idx}. {item['title']}**")
            st.markdown(item["answer"])
            if idx < len(st.session_state.history):
                st.divider()
