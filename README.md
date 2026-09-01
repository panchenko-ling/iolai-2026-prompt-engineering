# IOL-AI Challenge 2026 — Prompt Engineering Submission

Submission to the IOL-AI Challenge (Cohere Labs, July 2026), a competition in which language models solve International Linguistics Olympiad problems: self-contained puzzles that require inferring the grammar of an unfamiliar language from a small set of examples.

The entry placed 7th out of 48 submissions and received an Honorable Mention.

No fine-tuning was involved. The model is an unmodified [Qwen2.5-14B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ); all gains come from prompt construction and output parsing. The submitted checkpoint is on [Hugging Face](https://huggingface.co/offelia39/iolai-2026-qwen14b-awq-explained-rules).

## Contents

| File | Description |
|------|-------------|
| `script.py` | The full submitted inference script: prompt construction, generation, answer parsing, and explanation pass |

## Approach

**Task-type conditioning.** Each problem carries a `task_type` label (translation, fill-in-the-blanks, letter matching, text-to-number, number-to-text). A shared system prompt is combined with a type-specific instruction describing the expected answer format, plus a permissive default for unseen types.

**Phonetic transcription detection.** Some problems are written in phonetic transcription and expect transcriptions back; others use transcription as *input* and expect ordinary orthography or an English gloss. Instructing the model to answer in brackets when the target is not a transcription costs exact matches, so the detector separates two questions — does the problem use transcription notation, and is transcription the expected output — checking the query for non-phonetic targets and bracketed input forms before falling back to a check on the data.

**Over-generation guard.** The single largest scoring gain came from telling the model exactly how many answers to produce. Counting the items is harder than it looks: an earlier version searched only for numbered lines in the query and found nothing on roughly 45% of problems. The current version tries an explicit range in the instruction, then numbered lines in the query, then an unnumbered list following a colon, then numbered or lettered items in the context — and returns zero rather than asserting a wrong count, which leaves the guard disabled instead of misfiring.

**Answer parsing.** Output after the final `FINAL ANSWERS:` marker is split one answer per line, with heuristics to drop commentary lines, then truncated or padded to the expected item count so answer positions stay aligned with items.

**Explanation pass.** The competition's jury track required a short explanation on a majority of problems. Rather than asking for a reasoning trace, a second pass takes the already-finalised answers and asks for a *solution key* — the affix inventory, the ordered sound changes with their conditioning environments, the numeral base — since that is the form official IOL solutions take. This pass is byte-identical in its first stage to the answer-only version, so the automatic score is unaffected, and a failure in the explanation pass can never lose the answers.

## Notes

`build_messages(row)` is the single entry point for prompt construction, shared by the submission loop and any test harness, so the two cannot drift apart.

The script expects the competition's `test.csv` at `/tmp/data/` and the model weights in the working directory. Problem data is not included here: IOL problems are the property of the olympiad and its problem authors.

## License

MIT — see `LICENSE`. Applies to the code in this repository; the underlying model is licensed separately by its authors.

## Contact

Yulia Panchenko, panchenko.ling@gmail.com
