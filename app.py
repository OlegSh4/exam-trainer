from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import os, json, re, httpx
from pathlib import Path
from typing import Optional, List

app = FastAPI()

TICKETS_DIR = Path("tickets")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-medium-latest"

def read_ticket(filename: str) -> str:
    path = TICKETS_DIR / filename
    if not path.exists() or path.suffix not in (".txt", ".md"):
        raise HTTPException(404, "Ticket not found")
    return path.read_text(encoding="utf-8")

async def mistral(system: str, user: str, max_tokens: int = 1200, api_key: str = "") -> str:
    import asyncio
    key = api_key or os.environ.get("MISTRAL_API_KEY", "")
    if not key:
        raise HTTPException(500, "MISTRAL_API_KEY not set. Add it in the app settings.")
    delays = [5, 15, 30]  # retry delays in seconds
    for attempt, delay in enumerate(delays + [None]):
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                MISTRAL_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": MISTRAL_MODEL, "max_tokens": max_tokens,
                      "messages": [{"role": "system", "content": system},
                                    {"role": "user",   "content": user}]}
            )
        if r.status_code == 429:
            if delay is None:
                raise HTTPException(429, "Mistral rate limit. Подождите минуту и попробуйте снова.")
            retry_after = int(r.headers.get("retry-after", delay))
            await asyncio.sleep(retry_after)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

def json_from(text: str) -> dict:
    # Strip markdown code fences
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    # Remove illegal control characters except \n \r \t
    clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', clean)

    # Valid JSON escape characters after backslash
    VALID_ESCAPES = set('"\\/bfnrtu')

    def fix_strings(s):
        """
        Walk the raw text character by character.
        Inside JSON string values, fix:
          - bare newline/cr/tab  -> \\n \\r \\t
          - invalid \\X escapes   -> \\X  (double the backslash)
          - \' (unnecessary but valid in JS) -> just '
        """
        result = []
        in_string = False
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == '\\' and in_string:
                nxt = s[i + 1] if i + 1 < len(s) else ''
                if nxt == "'":
                    # \' is not a JSON escape — just emit the apostrophe
                    result.append("'")
                    i += 2
                elif nxt in VALID_ESCAPES:
                    # valid JSON escape — keep both chars
                    result.append(ch)
                    result.append(nxt)
                    i += 2
                else:
                    # invalid escape (e.g. \( \) \[ \, \. \: ) — double the backslash
                    result.append('\\\\')
                    i += 1   # leave nxt for next iteration
            elif ch == '"':
                in_string = not in_string
                result.append(ch)
                i += 1
            elif in_string and ch == '\n':
                result.append('\\n')
                i += 1
            elif in_string and ch == '\r':
                result.append('\\r')
                i += 1
            elif in_string and ch == '\t':
                result.append('\\t')
                i += 1
            else:
                result.append(ch)
                i += 1
        return ''.join(result)

    # 1. Try direct parse (works when Mistral outputs clean JSON)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    # 2. Fix strings and retry
    fixed = fix_strings(clean)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # 3. Extract outermost { } and retry
    m = re.search(r'\{[\s\S]*\}', clean)
    if m:
        try:
            return json.loads(fix_strings(m.group()))
        except json.JSONDecodeError:
            pass
    raise ValueError("Could not parse JSON from model response")


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/api/ticket_text")
def get_ticket_text(filename: str):
    text = read_ticket(filename)
    return {"text": text}


@app.get("/api/tickets")
def list_tickets():
    files = sorted(
        f.name for f in TICKETS_DIR.iterdir()
        if f.suffix in (".txt", ".md") and not f.name.startswith(".")
    )
    tickets = []
    for fn in files:
        content = (TICKETS_DIR / fn).read_text(encoding="utf-8")
        first_line = content.strip().splitlines()[0].lstrip("#").strip()
        tickets.append({"filename": fn, "title": first_line})
    return tickets


class GenerateRequest(BaseModel):
    ticket_file: str
    question_type: str
    history: Optional[List[dict]] = []

SYSTEM_EXAMINER = r"""Ты — строгий, но справедливый экзаменатор по математике/физике/профильному предмету.
У тебя есть эталонный текст билета с вопросами и ответами (включая формулы в LaTeX).
Твоя задача: проверить ПОНИМАНИЕ, а не зубрёжку — засчитывай правильный смысл, сказанный другими словами.
ВСЕГДА отвечай ТОЛЬКО валидным JSON без markdown-обёртки. Никакого текста вне JSON.
ПРАВИЛА JSON: все строковые значения — в одну строку без реальных переносов строк.
Формулы LaTeX ОБЯЗАТЕЛЬНО оборачивай в $...$ (инлайн) или $$...$$ (блочные). НИКОГДА не пиши LaTeX-команды (\frac \int \neg \Rightarrow и др.) голыми — только внутри $ $. Пиши только валидный LaTeX: \lim_{n\to\infty} а не \lim_{no\infty}, \to а не \rightarrow в индексах, проверяй скобки {}. НИКОГДА не используй ** или * внутри $$...$$ формул — только снаружи. В формулах-вариантах ответа избегай сложных вложенных \frac внутри \frac — упрощай запись.
Вопрос не длиннее 200 символов. Будь лаконичен."""

@app.post("/api/generate_question")
async def generate_question(req: GenerateRequest, request: Request):
    api_key = request.headers.get("X-Mistral-Key", "")
    ticket_text = read_ticket(req.ticket_file)

    history_summary = ""
    if req.history:
        history_summary = "\n\nПредыдущие вопросы в этой сессии (не повторяй их):\n"
        for h in req.history[-10:]:
            history_summary += f"- {h.get('question','')[:80]}\n"

    qtype = req.question_type
    if qtype == "mixed":
        import random
        qtype = random.choice(["open", "choice", "formula_match", "formula_choice"])
    if qtype == "final_open":
        pass  # handled below

    if qtype == "final_open":
        prompt = f"""Это финальный вопрос сессии. Задай ОДИН большой открытый вопрос, охватывающий ВЕСЬ билет целиком.
Вопрос должен требовать развёрнутого ответа по всем ключевым разделам билета.
Начни вопрос со слов "Расскажите о..." или "Объясните..." или "Опишите...".
Верни JSON:
{{
  "type": "open",
  "question": "текст вопроса по всему билету",
  "reference_answer": "полный эталонный ответ со всеми ключевыми пунктами",
  "hint": "перечисли разделы билета, которые нужно затронуть"
}}
БИЛЕТ:
{ticket_text}"""

    elif qtype == "open":
        prompt = f"""На основе билета сгенерируй ОДИН открытый вопрос по МАТЕМАТИЧЕСКОЙ или ТЕОРЕТИЧЕСКОЙ части билета.
Вопрос должен касаться определений, формул, теорем или доказательств из билета — без придуманных экономических примеров.
Верни JSON:
{{
  "type": "open",
  "question": "текст вопроса",
  "reference_answer": "эталонный ответ (для проверки, студент не видит)",
  "hint": "подсказка, намекающая на правильный ответ, но не раскрывающая его"
}}
{history_summary}
БИЛЕТ:
{ticket_text}"""

    elif qtype == "choice":
        prompt = f"""На основе билета сгенерируй вопрос с вариантами ответа (один правильный).
Верни JSON:
{{
  "type": "choice",
  "question": "текст вопроса",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "correct_index": 0,
  "explanation": "объяснение правильного ответа",
  "hint": "подсказка"
}}
{history_summary}
БИЛЕТ:
{ticket_text}"""

    elif qtype == "formula_choice":
        is_code = any(kw in ticket_text[:500] for kw in ["def ", "import ", "print(", "```python", "class ", ".py"])
        if is_code:
            prompt = f"""На основе билета про программирование создай вопрос с выбором: покажи 4 варианта кода/синтаксиса, один правильный.
НЕ используй LaTeX ($$). Пиши код как обычный текст в вариантах.
Верни JSON:
{{
  "type": "formula_choice",
  "question": "Какой вариант кода верен для [действие]?",
  "formula_name": "название операции",
  "options": ["вариант кода 1", "вариант кода 2", "вариант кода 3", "вариант кода 4"],
  "correct_index": 0,
  "explanation": "почему именно этот вариант верен",
  "hint": "подсказка"
}}
Остальные 3 варианта — правдоподобные ошибки (неверный метод, синтаксис, порядок аргументов).
{history_summary}
БИЛЕТ:
{ticket_text}"""
        else:
            prompt = f"""На основе билета возьми одну ключевую формулу и создай вопрос: покажи несколько похожих формул, одна из них правильная.
Верни JSON:
{{
  "type": "formula_choice",
  "question": "Какая из формул является верной записью [название]?",
  "formula_name": "название формулы",
  "options": ["$$формула1$$", "$$формула2$$", "$$формула3$$", "$$формула4$$"],
  "correct_index": 0,
  "explanation": "почему именно эта формула верна",
  "hint": "подсказка о ключевом элементе формулы"
}}
Остальные 3 формулы должны быть правдоподобными ошибками (неправильный знак, другой показатель степени и т.д.).
{history_summary}
БИЛЕТ:
{ticket_text}"""

    elif qtype == "formula_match":
        is_code = any(kw in ticket_text[:500] for kw in ["def ", "import ", "print(", "```python", "class ", ".py"])
        if is_code:
            prompt = f"""На основе билета про программирование создай задание: сопоставить метод/операцию с её результатом или описанием.
НЕ используй LaTeX ($$). Пиши методы и код как обычный текст — кратко, без обёрток.
Верни JSON:
{{
  "type": "formula_match",
  "question": "Сопоставьте метод с его описанием",
  "formulas": ["метод1()", "метод2()", "метод3()"],
  "labels": ["что делает 1", "что делает 2", "что делает 3"],
  "correct_order": [0, 1, 2],
  "explanation": "объяснение каждого соответствия",
  "hint": "подсказка"
}}
correct_order[i] = индекс label, соответствующего formulas[i].
Примеры formulas: ["d.keys()", "d.pop('a')", "d.get('x', 0)"]
Примеры labels: ["возвращает все ключи", "удаляет ключ и возвращает значение", "безопасно получает значение"]
{history_summary}
БИЛЕТ:
{ticket_text}"""
        else:
            prompt = f"""На основе билета возьми 3-4 формулы и создай задание: сопоставить формулу с её названием/смыслом.
Верни JSON:
{{
  "type": "formula_match",
  "question": "Сопоставьте формулы с их названиями",
  "formulas": ["$$f_1$$", "$$f_2$$", "$$f_3$$"],
  "labels": ["Название 1", "Название 2", "Название 3"],
  "correct_order": [0, 1, 2],
  "explanation": "объяснение каждого соответствия",
  "hint": "подсказка"
}}
correct_order[i] = индекс label, соответствующего formulas[i].
{history_summary}
БИЛЕТ:
{ticket_text}"""
    elif qtype not in ("open", "choice", "formula_choice", "formula_match", "final_open"):
        raise HTTPException(400, "Unknown question type")

    raw = await mistral(SYSTEM_EXAMINER, prompt, max_tokens=1500, api_key=api_key)
    try:
        data = json_from(raw)
        return data
    except Exception:
        pass

    # Response was truncated or malformed — ask model to redo it shorter
    retry_prompt = (
        "Предыдущий ответ был обрезан или содержал ошибки JSON. "
        "Сгенерируй БОЛЕЕ КОРОТКИЙ вопрос того же типа. "
        "ВАЖНО: вопрос и все строки должны быть краткими (не длиннее 120 символов). "
        "Отвечай ТОЛЬКО валидным JSON без markdown, без переносов строк внутри строк.\n\n"
        + prompt
    )
    raw2 = await mistral(SYSTEM_EXAMINER, retry_prompt, max_tokens=1500, api_key=api_key)
    try:
        data = json_from(raw2)
        return data
    except Exception as e:
        raise HTTPException(500, f"JSON parse error: {e}\nRaw: {raw2[:300]}")


class CheckRequest(BaseModel):
    ticket_file: str
    question: str
    question_type: str
    reference_answer: str
    user_answer: str

@app.post("/api/check_answer")
async def check_answer(req: CheckRequest, request: Request):
    api_key = request.headers.get("X-Mistral-Key", "")
    prompt = f"""Оцени ответ студента на вопрос экзамена.
Вопрос: {req.question}
Эталонный ответ: {req.reference_answer}
Ответ студента: {req.user_answer}

Засчитывай правильный смысл, выраженный другими словами. Частичный балл возможен.
Верни JSON:
{{
  "score": 0-100,
  "verdict": "correct | partial | wrong",
  "feedback": "детальный разбор ответа студента",
  "missing": "что упущено или неверно (если есть)"
}}"""
    raw = await mistral(SYSTEM_EXAMINER, prompt, max_tokens=600, api_key=api_key)
    try:
        return json_from(raw)
    except Exception as e:
        raise HTTPException(500, f"JSON parse error: {e}\nRaw: {raw[:300]}")


class HintRequest(BaseModel):
    ticket_file: str
    question: str
    hint: str
    attempt: int = 1

@app.post("/api/get_hint")
async def get_hint(req: HintRequest, request: Request):
    api_key = request.headers.get("X-Mistral-Key", "")
    prompt = f"""Студент затрудняется с вопросом: {req.question}
Базовая подсказка: {req.hint}
Номер попытки получить подсказку: {req.attempt}

Дай подсказку уровня {req.attempt} (1=лёгкий намёк, 2=более конкретный намёк, 3=почти ответ).
Верни JSON: {{"hint": "текст подсказки"}}"""
    raw = await mistral(SYSTEM_EXAMINER, prompt, max_tokens=300, api_key=api_key)
    try:
        return json_from(raw)
    except:
        return {"hint": req.hint}


class SummaryRequest(BaseModel):
    ticket_file: str
    history: List[dict]

@app.post("/api/summary")
async def get_summary(req: SummaryRequest, request: Request):
    api_key = request.headers.get("X-Mistral-Key", "")
    ticket_text = read_ticket(req.ticket_file)
    history_text = json.dumps(req.history, ensure_ascii=False, indent=2)
    prompt = f"""Студент прошёл экзаменационную сессию. Вот история вопросов и ответов:
{history_text}

Билет:
{ticket_text[:1500]}

Дай итоговый анализ. Верни JSON:
{{
  "overall_score": 0-100,
  "grade": "Отлично/Хорошо/Удовлетворительно/Не сдал",
  "strong_areas": ["тема1", "тема2"],
  "weak_areas": ["тема3", "тема4"],
  "recommendation": "что подучить и как",
  "summary_text": "2-3 предложения общего вывода"
}}"""
    raw = await mistral(SYSTEM_EXAMINER, prompt, max_tokens=800, api_key=api_key)
    try:
        return json_from(raw)
    except Exception as e:
        raise HTTPException(500, f"JSON parse error: {e}\nRaw: {raw[:300]}")


# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("static/index.html")
