"""Book translation prompt construction and chunk translation."""

import logging
from pathlib import Path
from typing import Dict, Optional

from core.exceptions import TranslationError
from infra.llm.client import OpenAIClient

logger = logging.getLogger(__name__)
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "translation.txt"


DEFAULT_TRANSLATION_PROMPT = """You are a professional book translator. You are translating a book from {source_lang} to {target_lang}.
Instructions:
- Translate the text accurately and naturally.
- Do not summarize or omit any content.
- Do not add commentary.
- Do not translate code snippets, formulas, or other text that should remain unchanged.
- Preserve paragraph breaks as in the original text.
- Maintain the tone, style, and structure of the original.
- Keep inline markers like **bold** and *italic* untouched.
{glossary}
Text to translate:
-----
{text}
-----
Now output only the translated text."""


def load_translation_prompt_template() -> str | None:
    """Load the default translation prompt template or return None to use the built-in default."""
    if not DEFAULT_PROMPT_PATH.exists():
        logger.debug("Translation prompt template not found at %s; using default template", DEFAULT_PROMPT_PATH)
        return None

    logger.debug("Loading translation prompt template from %s", DEFAULT_PROMPT_PATH)
    return DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")


def format_glossary(glossary: Optional[Dict[str, str]]) -> str:
    """Format an optional term glossary for insertion into the prompt."""
    if not glossary:
        logger.debug("No glossary provided for translation prompt")
        return ""

    glossary_entries = "\n".join(f"{source} -> {target}" for source, target in glossary.items())
    logger.debug("Formatted glossary with %s entries", len(glossary))
    return (
        "\nHere is a glossary of terms and how they should be translated:\n"
        f"{glossary_entries}\n"
    )


def build_translation_prompt(
    text: str,
    glossary: Optional[Dict[str, str]],
    source_lang: str,
    target_lang: str,
    prompt_template: str | None = None,
) -> str:
    """Build the final user prompt for one translation chunk."""
    template = prompt_template or DEFAULT_TRANSLATION_PROMPT
    logger.debug(
        "Building translation prompt. source_lang=%s target_lang=%s text_chars=%s custom_template=%s",
        source_lang,
        target_lang,
        len(text),
        prompt_template is not None,
    )
    return template.format(
        source_lang=source_lang,
        target_lang=target_lang,
        glossary=format_glossary(glossary),
        text=text,
    )


class OpenAIBookTranslator:
    """Translate book text chunks with an OpenAI-compatible chat model."""

    def __init__(
        self,
        llm_client: OpenAIClient | None = None,
        model_name: str | None = None,
        prompt_template: str | None = None,
    ):
        """Create a chunk translator with optional injected client and prompt."""
        logger.debug(
            "Initializing OpenAIBookTranslator. model_name=%s custom_client=%s custom_prompt_template=%s",
            model_name,
            llm_client is not None,
            prompt_template is not None,
        )
        self.client = llm_client or OpenAIClient(model_name=model_name)
        self.model_name = model_name or self.client.model_name
        self.prompt_template = prompt_template

    def translate_chunk(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: Optional[Dict[str, str]] = None,
    ) -> str:
        """Translate one text chunk and return only the translated content."""
        logger.debug(
            "Translating chunk. source_lang=%s target_lang=%s chars=%s glossary_entries=%s",
            source_lang,
            target_lang,
            len(text),
            len(glossary or {}),
        )
        prompt = build_translation_prompt(
            text=text,
            glossary=glossary,
            source_lang=source_lang,
            target_lang=target_lang,
            prompt_template=self.prompt_template,
        )

        try:
            response = self.client.create_chat_completion(
                model_name=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional literary translator from {source_lang} to {target_lang}.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=8000,
            )
        except Exception as exc:
            logger.exception("Chunk translation failed")
            raise TranslationError(f"Failed to translate chunk: {exc}") from exc

        if response.choices[0].finish_reason == "length":
            logger.warning("Translation truncated due to max_tokens limit")

        logger.debug("Chunk translation completed. finish_reason=%s", response.choices[0].finish_reason)
        return response.choices[0].message.content.strip()
