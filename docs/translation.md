# Translating the Olist reviews to English

The Olist reviews are Portuguese. Search, the UI and the tutorial are easier to follow in English, so `make translate`
(`src/01b_translate_dataset.py`) builds an English copy of the reviews file. This page records how the model was chosen
and what the translation does and does not guarantee.

## Model comparison

I translated the same 52 real reviews (20 short, 20 medium, 10 long, plus 2 known troublemakers) on a CPU with four free models and read
the outputs. This is a small, informal check, not a benchmark.

| Model | Texts/sec (torch, 4 threads, busy laptop) | What went wrong |
|---|---:|---|
| `Helsinki-NLP/opus-mt-ROMANCE-en` | 6.4 | Short reviews turned into unrelated text (`rapidez na entrega.` -> a sentence about an EU directive); emoji became a long run of dots; sentences repeated ("the price of the price of ..."). |
| `Helsinki-NLP/opus-mt-mul-en` | 31 | Fast, but garbled many reviews (`Gostei, som mt. bom.` -> "You're welcome."), looped on long ones, dropped sentences. |
| `facebook/nllb-200-distilled-600M` | 4.1 | Often fluent, but sometimes invents content (`RAPIDEZ E EFICIÊNCIA.` -> "The results of the study were published in the Official Journal of the European Union."), and drops sentences. |
| `unicamp-dl/translation-pt-en-t5` | 4.1 | Closest to the source on nearly all reviews and never invented unrelated text. Weak spots: very short fragments (`ate proxima` -> "approach."), ALL-CAPS and emoji-only text, and it drops sentences from long inputs. The last three are handled by guards in `src/translation.py`. |

**Chosen:** `unicamp-dl/translation-pt-en-t5`, converted to CTranslate2 int8, which ran at about 18 texts/sec on the same 52 reviews
(about 4x faster than PyTorch) with the same quality. On the full dataset the 39,074 distinct texts take roughly 30 to 40 minutes.

## What the code adds around the model

- Long reviews are split into sentences and translated piece by piece (merging tiny pieces so `mt. bom` stays together).
- ALL-CAPS text is lower-cased first; runs like `........` are collapsed.
- Text with no letters (emoji, digits) is kept as is.
- A translation that is empty or much longer than its input is rejected and the original text is kept; the job logs how many.
- A period is added to inputs without final punctuation and removed afterwards (otherwise short inputs get junk appended).

## Limits

- It is machine translation. Misspellings can become wrong words (`otino` -> "otitis"), idioms get translated literally, and the dataset's anonymised store name appears as "lannister".
- Embedding models capture overall meaning, so search quality is only mildly affected, but any single translated sentence can be off.
- The translated file is generated locally and is not committed (the Olist data is licensed CC BY-NC-SA 4.0).
