# VARIANT: tightened phonetic detection + robust item counting + JURY EXPLANATIONS
# Two passes per problem. Pass 1 is byte-identical to the answer-only version, so the
# automatic score is unaffected. Pass 2 takes the FINISHED answers and asks for a short
# structured explanation -- the rules ask for "a short explanation of the answer", NOT a
# raw reasoning trace, and require a valid explanation on a majority of problems.
# The explanation prompt asks for a solution KEY -- the rule system -- because that is
# the form official IOL solutions take (affix inventories, ordered sound changes with
# their environments, numeral bases), not a walkthrough of individual answers.
# The over-generation guard (the single biggest scoring gain) only fires when the item
# count is known. The old regex looked for numbered lines in the query alone, which
# found nothing on ~45% of problems -- ranges like "(1-8)", unnumbered lists, and
# match_letters problems whose items live in the context all counted as zero.
# v3: the detector now checks the QUERY, not just the context. If the query itself
# contains bracketed forms, those are the task's INPUT (convert FROM transcription),
# so the answers should not be transcriptions and the instruction must not fire.
# Prompt construction is exposed via build_messages(row) -- the single entry point
# shared by this script and any test harness.

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
MODEL_ID = "."
MAX_NEW_TOKENS = 2048   # 5 problems / 30 min = generous budget; experiment freely

import re
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto",
).eval()

SYSTEM_BASE = (
    "You solve International Linguistics Olympiad problems by reasoning from the "
    "data you are given. You may meet a task type you have never seen: read the "
    "instruction and the examples, and answer in the same form they use. "
    "Reason step by step first. Then write a line that says exactly FINAL ANSWERS: "
    "and, below it, one answer per line in the order the items are asked -- the "
    "bare answer only, no numbering, no quotes, no extra text. "
    "After FINAL ANSWERS:, output only the answers, exactly one line per numbered "
    "item, then stop -- no commentary, extra lines, or repeated answers."
)

# Per-task-type instruction, injected based on the row's task_type column.
TASK_INSTRUCTIONS = {
    "translation": (
        "This is a TRANSLATION task. Give only the translated form, in the language "
        "the task asks for. No explanation, no source form, just the translation."
    ),
    "fill_blanks": (
        "This is a FILL-IN-THE-BLANKS task. Give only the missing form for each blank, "
        "nothing else."
    ),
    "match_letters": (
        "This is a MATCHING task. Each numbered item must be answered with a SINGLE "
        "OPTION LETTER only (for example: C). Do NOT write the word, meaning, or "
        "translation -- only the letter that matches. If item 1 matches option C, the "
        "answer for item 1 is exactly: C"
    ),
    "text_to_num": (
        "This is a TEXT-TO-NUMBER task. Give the number in digits only (for example: 111)."
    ),
    "num_to_text": (
        "This is a NUMBER-TO-TEXT task. Write the number out in words, in the language "
        "the task asks for. Give only the written-out form."
    ),
}

# Default for task types not in the dict (the live set may include others).
TASK_DEFAULT = (
    "Give exactly what the instruction asks for, in the same form the examples use, "
    "and nothing else."
)

# Appended when the problem's examples are written as phonetic transcriptions.
PHONETIC_INSTRUCTION = (
    "IMPORTANT -- this problem uses PHONETIC TRANSCRIPTION. The examples write forms "
    "inside square brackets, like [bø:va]. Your answers must be phonetic transcriptions "
    "in exactly that same notation: enclosed in square brackets, using the same phonetic "
    "symbols, length marks (:), and diacritics that the examples use. Do NOT give an "
    "English meaning, gloss, or translation -- give the transcribed FORM. If the examples "
    "write [kno:ar], an answer must look like [kno:ar], not like a word such as 'kneads'."
)

# Characters that suggest phonetic notation rather than ordinary orthography.
_IPA_HINT = re.compile(
    r"[\u0250-\u02AF"      # IPA Extensions (ɐ ɔ ə ɟ ʃ ʌ ...)
    r"\u02B0-\u02FF"       # spacing modifier letters (ʰ ʲ ˈ ˌ ː ...)
    r"\u0300-\u036F"       # combining diacritics (tone marks, etc.)
    r"\u1D00-\u1D7F"       # phonetic extensions
    r"øœæðθŋɣʔ]"           # common non-IPA-block phonetic characters
)

def _bracketed_forms(text):
    """Return the contents of [...] groups that look like transcriptions, not citations."""
    out = []
    for m in re.finditer(r"\[([^\[\]\n]{1,40})\]", text):
        inner = m.group(1).strip()
        if not inner:
            continue
        # Skip bracketed numbers/citations like [1] or [see 3].
        if re.fullmatch(r"[\d\s,.\-]+", inner):
            continue
        out.append(inner)
    return out

# The query asks for a NON-transcription output (so brackets in the data are inputs).
_ASKS_NON_PHONETIC = re.compile(
    r"(?i)translate\s+into\s+english"
    r"|write\s+(it\s+)?in\s+the\s+[\w'\u2019-]+\s+orthography"
    r"|in\s+the\s+regular\s+orthography"
)

# The query explicitly asks for a transcription (overrides the bracket-in-query check).
_ASKS_TRANSCRIPTION = re.compile(r"(?i)\b(transcribe|transcription|phonetic(ally)?)\b")

def is_phonetic_task(context, query, min_forms=3):
    """Heuristic: should the ANSWERS be bracketed phonetic transcriptions?

    Two separate questions, and earlier versions conflated them:
      (a) does the problem *use* transcription notation?  -- look at the data
      (b) is transcription the *expected output*?         -- look at the query

    A problem can be full of [brackets] while asking for ordinary orthography or an
    English gloss; in that case the brackets are the task's INPUT and telling the model
    to answer in brackets actively costs exact matches. So:

      - reject outright when the query names a non-phonetic target
        ("translate into English", "write in the X orthography")
      - reject when the query itself contains bracketed forms (they are the items to
        convert FROM), unless the query also explicitly asks to transcribe
      - otherwise fall back to the data check: several bracketed forms, enough of them
        containing IPA symbols, length marks, or diacritics
    """
    if _ASKS_NON_PHONETIC.search(query):
        return False
    if _bracketed_forms(query) and not _ASKS_TRANSCRIPTION.search(query):
        return False

    forms = _bracketed_forms(context) + _bracketed_forms(query)
    if len(forms) < min_forms:
        return False
    phonetic_looking = sum(1 for f in forms if _IPA_HINT.search(f) or ":" in f)
    return phonetic_looking >= max(2, len(forms) // 4)

def build_system(task_type, context="", query=""):
    """Shared base + the instruction for this task_type + phonetic note when detected."""
    specific = TASK_INSTRUCTIONS.get(str(task_type).strip().lower(), TASK_DEFAULT)
    parts = [SYSTEM_BASE, specific]
    if is_phonetic_task(context, query):
        parts.append(PHONETIC_INSTRUCTION)
    return "\n\n".join(parts)

_RANGE_RE = re.compile(r"\((\d+)\s*[-\u2010-\u2015\u2212]\s*(\d+)\)")

def _count_numbered(text):
    """Lines beginning with a number marker: '1.' '2)' '17.'"""
    return len(re.findall(r"(?m)^\s*\d+[.)]", text))

def _count_lettered(text):
    """Lines beginning with a letter marker: 'A.' 'B)' 'N.'"""
    return len(re.findall(r"(?m)^\s*[A-Z][.)]\s", text))

def _count_list_lines(text):
    """Count the item lines of an unnumbered list following an instruction line.

    "Translate into Sursilvan:\n\nelms\nangles" -> 2. Takes everything after the first
    line ending in a colon and counts the non-empty lines after it. Lines containing
    '|' are skipped: those are table data, not items.
    """
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.rstrip().endswith(":"):
            start = i + 1
            break
    if start is None:
        return 0
    return len([ln for ln in lines[start:] if ln.strip() and "|" not in ln])

def count_items(query, context=""):
    """Best-effort count of how many answers the problem expects.

    The over-generation guard only fires when this returns > 0. Four sources, tried in
    order of how explicit they are:

      1. an explicit range in the instruction, "Fill in the blanks (1-8)"  -> 8
      2. numbered lines in the query, "1. foo\n2. bar"                     -> 2
      3. an unnumbered list after a colon, "Translate into X:\n a\n b"     -> 2
      4. numbered or lettered items in the CONTEXT, for queries such as
         "Determine the correct correspondences." that carry no items at all

    Returns 0 when nothing plausible is found, which leaves the guard disabled rather
    than asserting a wrong count.
    """
    m = _RANGE_RE.search(query)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if hi >= lo and hi - lo < 200:
            return hi - lo + 1

    n = _count_numbered(query)
    if n:
        return n

    n = _count_list_lines(query)
    if n:
        return n

    if context:
        n = _count_numbered(context)
        if n:
            return n
        n = _count_lettered(context)
        if n:
            return n

    return 0

def _looks_like_prose(line):
    """True if the line reads as commentary rather than an answer."""
    if (
        re.search(r"(?i)^(final answers?|answers?|note|reviewing|summary|explanation)\b.*:$", line)
        or re.search(r"(?i)^(here (are|is)|the (final )?answers? (are|is)|based on|therefore|thus|in summary)\b", line)
        or re.search(r"(?i)^(these|this) (translations?|answers?|follow|matches)\b", line)
    ):
        return True
    # A line ending in a colon is almost always a heading, not an answer.
    if line.rstrip().endswith(":"):
        return True
    return False

def parse_answers(text, n_items=0):
    """Keep only the lines after the last 'FINAL ANSWERS:' marker, one answer per line.

    Then enforce the expected item count:
      - drop lines that read as commentary rather than answers
      - truncate to n_items
      - pad with "" if the model produced fewer than n_items, so positions stay aligned
    n_items <= 0 disables count enforcement (falls back to old behaviour).
    """
    marker = list(re.finditer(r"(?im)^\s*final answers?\s*:?\s*$", text))
    if marker:
        text = text[marker[-1].end():]

    answers = []
    for line in text.splitlines():
        line = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
        if not line:
            continue
        if _looks_like_prose(line):
            continue
        answers.append(line)

    if n_items > 0:
        answers = answers[:n_items]
        if len(answers) < n_items:
            answers += [""] * (n_items - len(answers))
    return answers

def build_user(row):
    """User message: the problem, plus the over-generation guard stating the item count."""
    n_items = count_items(row["query"], row["context"])
    content = f"{row['context'].strip()}\n\n{row['query'].strip()}"
    if n_items > 0:
        content += (
            f"\n\nThere are exactly {n_items} items to answer. "
            f"Give exactly {n_items} answers after FINAL ANSWERS:, "
            "one per line, no more and no fewer."
        )
    return content

def build_messages(row):
    """THE single entry point for prompt construction.

    Any harness (the submission loop below, or a test notebook) should call only this,
    so changes to prompt logic, function signatures, or module constants cannot cause
    the two to drift apart.
    """
    return [
        {"role": "system", "content": build_system(row["task_type"], row["context"], row["query"])},
        {"role": "user", "content": build_user(row)},
    ]

# ---------------------------------------------------------------------------
# Pass 2: jury explanations
# ---------------------------------------------------------------------------

EXPLANATION_SYSTEM = (
    "You are writing the solution key for an International Linguistics Olympiad problem, "
    "for expert linguists. Official solution keys state the SYSTEM that generates the "
    "answers, not a walkthrough of individual items.\n\n"
    "State the rules. Depending on the problem that means: the affixes or morphemes and "
    "what each one marks; the sound changes, written as rules with their conditioning "
    "environment (for example 'k becomes g after a nasal', 'the second vowel lengthens "
    "when both are short'); the word order or constructions; and any numeral system or "
    "counting base. If several rules compete, say which applies first.\n\n"
    "Write the rules compactly -- a short list, or a table of form and meaning. Cite the "
    "actual forms from the data as evidence for each rule. Do NOT walk through the "
    "answers one by one, do NOT restate the answer list, and do NOT narrate your "
    "reasoning process. Do not hedge or apologise. Keep it under 200 words."
)

MAX_EXPLANATION_TOKENS = 600
EXPLANATION_CHAR_LIMIT = 2000

def build_explanation_messages(row, answers):
    """Ask for a short, structured account of answers that have already been decided."""
    answer_block = "\n".join(
        f"{i}. {a}" for i, a in enumerate(answers, 1) if str(a).strip()
    )
    return [
        {"role": "system", "content": EXPLANATION_SYSTEM},
        {"role": "user", "content": (
            f"{row['context'].strip()}\n\n{row['query'].strip()}\n\n"
            f"The answers given were:\n{answer_block}\n\n"
            "Write the solution key: state the rules of this language that the data "
            "reveals and that produce these answers."
        )},
    ]

def clean_explanation(text):
    """Tidy the generated explanation: strip any stray answer-block marker and clamp length."""
    marker = list(re.finditer(r"(?im)^\s*final answers?\s*:?\s*$", text))
    if marker:
        text = text[:marker[0].start()]
    text = text.strip()
    if len(text) > EXPLANATION_CHAR_LIMIT:
        cut = text[:EXPLANATION_CHAR_LIMIT]
        nl = cut.rfind("\n")
        text = (cut[:nl] if nl > EXPLANATION_CHAR_LIMIT // 2 else cut).rstrip()
    return text

def generate_explanation(row, answers):
    """Second generation pass. Never raises: a failed explanation must not lose the answers."""
    try:
        if not any(str(a).strip() for a in answers):
            return ""
        enc = tok.apply_chat_template(
            build_explanation_messages(row, answers),
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_EXPLANATION_TOKENS, do_sample=False)
        text = tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        return clean_explanation(text)
    except Exception as exc:
        print(f"  explanation failed: {exc}", flush=True)
        return ""

df = pd.read_csv("/tmp/data/test.csv", dtype=str).fillna("")

rows = []
for _, r in df.iterrows():
    n_items = count_items(r["query"], r["context"])
    messages = build_messages(r)
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    text = tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    answers = parse_answers(text, n_items)

    # Pass 2: explain the answers we just settled on. This cannot change `answers`.
    explanation = generate_explanation(r, answers)

    rows.append({
        "id": r["id"],
        "pred": json.dumps(answers, ensure_ascii=False),
        "explanation": explanation,
    })
    print(f"{len(rows)}/{len(df)} done "
          f"({len(explanation)} chars of explanation)", flush=True)

_n_expl = sum(1 for row in rows if row["explanation"].strip())
print(f"explanations present on {_n_expl}/{len(rows)} problems "
      f"({100 * _n_expl / max(len(rows), 1):.0f}%; jury track needs >50%)", flush=True)

pd.DataFrame(rows).to_csv("submission.csv", index=False)
print("wrote submission.csv", flush=True)
