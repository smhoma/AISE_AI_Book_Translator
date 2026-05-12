# Book Translator

Book Translator is a PDF translation tool. It converts scanned or image-based PDF pages into images, extracts text with Tesseract OCR, rebuilds paragraphs, removes repeated headers and footers, chunks the text for an LLM, translates each chunk with an OpenAI-compatible chat completion API, and saves the translated book as Markdown, HTML, and PDF.

The project also includes a simple Gradio GUI for uploading an input PDF and downloading the translated output files.

## Current Status

The active application is organized around these entrypoints:

- `main.py`: Gradio GUI for uploading a PDF, running translation, and downloading the Markdown, HTML, and PDF outputs.
- `app/use_cases/translate_book.py`: main translation orchestration use case and output writer.
- `app/use_cases/translate_chunks.py`: book-specific prompt construction and chunk translation.

The `archive/` folder is historical reference code and is not used by the active application.

## Project Structure

```text
.
|-- main.py
|-- requirements.txt
|-- sample.env
|-- app/
|   |-- prompts/
|   |   `-- translation.txt
|   `-- use_cases/
|       |-- translate_book.py
|       `-- translate_chunks.py
|-- core/
|   |-- config.py
|   |-- exceptions.py
|   |-- logging_config.py
|   |-- output_rendering.py
|   `-- text_processing.py
|-- infra/
|   |-- llm/
|   |   `-- client.py
|   `-- ocr/
|       |-- pdf.py
|       `-- tesseract.py
|-- input/
|-- output/
`-- archive/
```

## How The Pipeline Works

1. The user uploads a PDF through the Gradio GUI.
2. `main.py` saves the uploaded file into `input/`.
3. `BookTranslator.translate_pdf()` creates temporary page images from the PDF.
4. `infra/ocr/pdf.py` uses `pdf2image` to render PDF pages.
5. `infra/ocr/tesseract.py` uses Tesseract TSV output to extract words and rebuild paragraphs.
6. `core/text_processing.py` detects repeated headers and footers, removes them, merges paragraphs across page boundaries, and chunks text.
7. `app/use_cases/translate_chunks.py` builds the translation prompt and sends each chunk through the configured OpenAI-compatible client.
8. The translated chunks are merged.
9. `app/use_cases/translate_book.py` formats Markdown and HTML, writes all translated outputs, and renders the translated PDF.
10. The final Markdown, HTML, PDF, and raw translation text files are saved in `output/`.
11. Gradio returns the generated user-facing files as downloadable outputs.

## Main Components

### `main.py`

The GUI entrypoint. It exposes:

- PDF upload.
- Source language.
- Target language.
- Translate button.
- Downloadable translated Markdown, HTML, and PDF files.
- Status output.

The GUI intentionally hides these advanced settings:

- OCR language code.
- Max tokens per chunk.
- Model name.

Those values are still used internally from `core.config.settings`.

### `app/use_cases/translate_book.py`

This is the application layer. It coordinates the full workflow:

- PDF conversion.
- OCR.
- Paragraph cleanup.
- Chunking.
- Chunk translation.
- Output artifact creation.

The main class is `BookTranslator`.

### `app/use_cases/translate_chunks.py`

Owns book-specific LLM translation behavior:

- Default prompt template loading.
- Glossary formatting.
- Prompt construction.
- Chunk translation.

### `core/config.py`

Loads runtime configuration from environment variables and `.env`.

Important settings:

| Variable | Default | Purpose |
|---|---:|---|
| `API_KEY` or `OPENAI_API_KEY` | none | API key for the OpenAI-compatible provider. |
| `BASE_URL` | none | Optional custom OpenAI-compatible base URL. |
| `MODEL_NAME` | `gpt-4o` | Translation model. |
| `INPUT_DIR` | `./input` | Folder where uploaded PDFs are stored. |
| `OUTPUT_DIR` | `./output` | Folder where translated files are written. |
| `SOURCE_LANG` | `English` | Default source language. |
| `TARGET_LANG` | Persian | Default target language. |
| `OCR_LANG_CODE` | `eng` | Tesseract language code. |
| `MAX_TOKENS` | `2500` | Approximate max tokens per translation chunk. |
| `PDF_RENDERER_PATH` | auto-detect | Optional Chrome/Chromium executable for PDF rendering. |
| `LOG_LEVEL` | `DEBUG` | Project logging level. |

### `core/text_processing.py`

Contains the text cleanup and chunking logic:

- Header and footer normalization.
- Repeated header/footer detection.
- Header/footer removal.
- Paragraph indentation heuristics.
- Cross-page paragraph merging.
- Rough token estimation.
- Sentence fallback chunking.
- Target-language direction detection.
- Markdown and HTML formatting.

### `core/output_rendering.py`

Contains output rendering helpers:

- Chrome/Chromium PDF renderer autodetection.
- HTML-to-PDF rendering.

### `infra/ocr/`

OCR infrastructure:

- `pdf.py`: PDF page rendering through `pdf2image`.
- `tesseract.py`: Tesseract TSV OCR and paragraph reconstruction.

This layer depends on both Python packages and system-level OCR tools.

### `infra/llm/`

LLM infrastructure:

- `client.py`: OpenAI-compatible client wrapper and provider retry handling.

### `app/prompts/`

Prompt templates:

- `translation.txt`: fixed prompt template used by the book translator.

The translation prompt supports these placeholders:

```text
{source_lang}
{target_lang}
{glossary}
{text}
```

## Setup

The project has been verified with the existing conda environment named `AI313`.

Run commands from the project root:

```bash
cd /home/mahdi/Codes/AI_Software_Engineer/session06_project1
conda activate AI313
```

If you prefer not to activate the shell environment:

```bash
conda run --no-capture-output -n AI313 python main.py
```

## Python Dependencies

Python dependencies are listed in `requirements.txt`:

```text
gradio
openai
python-dotenv
pdf2image
pytesseract
pillow
nltk
```

If you need to install them into another environment:

```bash
pip install -r requirements.txt
```

## System Dependencies

PDF rendering and OCR require system tools:

- Tesseract OCR.
- Tesseract language data for the OCR language code you use.
- Poppler utilities, required by `pdf2image`.

On Debian or Ubuntu-style systems, the packages are typically:

```bash
sudo apt install tesseract-ocr tesseract-ocr-eng poppler-utils
```

For other OCR languages, install the matching Tesseract language package and set `OCR_LANG_CODE`.

## Environment Configuration

Create a local `.env` file from `sample.env`:

```bash
cp sample.env .env
```

Then set your own values:

```env
API_KEY=your_api_key_here
BASE_URL=https://your-openai-compatible-provider/v1
MODEL_NAME=gpt-4o
LOG_LEVEL=DEBUG
SOURCE_LANG=English
TARGET_LANG=Persian
OCR_LANG_CODE=eng
MAX_TOKENS=2500
```

Do not commit `.env`; it should contain private credentials.

## Running The GUI

Start the Gradio application:

```bash
conda run --no-capture-output -n AI313 python main.py
```

By default, the app lets Gradio use its default port. To hard-code another port, edit `main.py`:

```python
GRADIO_SERVER_PORT: int | None = 10001
```

When running successfully, Gradio prints a local URL such as:

```text
http://127.0.0.1:7960
```

## Using The GUI

1. Open the local Gradio URL in a browser.
2. Upload a PDF.
3. Set the source language.
4. Set the target language.
5. Click `Translate`.
6. Download the translated Markdown file from the output component.

Uploaded PDFs are copied to `input/`.

Translated outputs are saved to `output/`.

## Output Files

For an input file named:

```text
my_book.pdf
```

The project writes:

```text
output/translated_my_book.md
output/translated_my_book.html
output/translated_my_book.pdf
output/translated_my_book.txt
```

The Markdown file is the main editable output. The HTML and PDF files are rendered outputs with explicit document direction metadata.

The raw text file is useful for debugging translation chunk boundaries and model output before Markdown formatting.

## Programmatic Usage

Example:

```python
from app.use_cases.translate_book import BookTranslator

translator = BookTranslator()
result = translator.translate_pdf(
    pdf_path="input/my_book.pdf",
    source_lang="English",
    target_lang="Persian",
)

print(result.output_path)
```

The returned `TranslationResult` includes:

- `output_path`
- `raw_output_path`
- `html_output_path`
- `pdf_output_path`
- `markdown_text`
- `html_text`
- `raw_text`
- `text_direction`
- `page_count`
- `paragraph_count`
- `chunk_count`

## Logging

Project logging is configured in `core/logging_config.py`.

Default logging level is `DEBUG`.

The format is:

```text
timestamp | level | logger name | message
```

Example:

```text
2026-05-06 14:17:04 | DEBUG    | __main__ | Launching Gradio using the default port
```

Set a different level with:

```bash
LOG_LEVEL=INFO conda run --no-capture-output -n AI313 python main.py
```

Third-party libraries such as `httpx`, `httpcore`, `urllib3`, `PIL`, `gradio`, and `asyncio` are reduced to warning level to keep project logs readable.

## Error Handling

Project-specific exceptions are defined in `core/exceptions.py`:

- `BookTranslatorError`
- `ConfigurationError`
- `FileProcessingError`
- `OCRProcessingError`
- `TranslationError`

The Gradio GUI catches exceptions, logs the stack trace, and shows a short error message in the status box.

## Known Limitations

- OCR quality depends heavily on PDF scan quality.
- Tesseract language data must match `OCR_LANG_CODE`.
- `estimate_tokens()` is a rough character-based estimator, not a tokenizer from the target model.
- Very large PDFs can take a long time because OCR and LLM translation are sequential.
- The GUI currently exposes only the main settings. OCR language, model, and chunk size are controlled through environment variables.

## Troubleshooting

### `API_KEY or OPENAI_API_KEY is not set`

Create `.env` and set `API_KEY` or `OPENAI_API_KEY`.

### `Failed to convert PDF to images`

Install Poppler utilities and confirm the PDF path exists.

### `Failed to OCR page image`

Install Tesseract and the required language package.

For English OCR, confirm `tesseract-ocr-eng` is installed and `OCR_LANG_CODE=eng`.

### Gradio cannot find an empty port

Hard-code a different port in `main.py`:

```python
GRADIO_SERVER_PORT: int | None = 10001
```

### NLTK tokenizer data is missing

The code falls back to a simple regex sentence splitter. For better sentence splitting, install the NLTK punkt data in the environment.

### `No PDF renderer found`

Install Google Chrome or Chromium, or set `PDF_RENDERER_PATH` to the executable path.

## Development Notes

- Keep orchestration in `app/use_cases/`.
- Keep reusable text and configuration logic in `core/`.
- Keep external integrations in `infra/`.
- Keep prompt templates in `app/prompts/`.
- Keep test PDFs in `input/`.
- Keep generated translations in `output/`.
- Do not modify `archive/` unless intentionally preserving or comparing historical versions.
