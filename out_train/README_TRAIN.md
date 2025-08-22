# Training JSONL Outputs

Generated from: `out\onepiece_chapters.jsonl`
Context window: 8

Files:
- train_next_beats.[train|val|test].jsonl
- train_synopsis_from_beats.[train|val|test].jsonl
- train_title_sugg.[train|val|test].jsonl
- train_thread_resolution.[train|val|test].jsonl
- multitask_mix.[train|val|test].jsonl
- rag_corpus.jsonl
- splits/train.ids.txt, val.ids.txt, test.ids.txt

## Fine-tuning formats
Each JSONL line is:
{"prompt": "...","completion":"...","chapter_id": N}

- For OpenAI fine-tuning: feed the files directly (rename to your liking).
- For HuggingFace SFT (trl): map to a single text field:
  prompt + "\n\n### ANSWER\n" + completion

## RAG
`rag_corpus.jsonl` has `id`, `title`, `chapter_id`, `arc`, `text`. Embed `text` and retrieve by `chapter_id` proximity for forecasting.

## License
Content derived from CC BY-SA summaries → your outputs inherit CC BY-SA. Keep attribution in your data pipeline.