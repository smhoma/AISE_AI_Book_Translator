import pytesseract
from pdf2image import convert_from_path
from typing import List, Tuple
import re
from openai import OpenAI
import os
from dotenv import load_dotenv

class BookTranslator:
    def __init__(self, source_lang: str = "en", target_lang: str = "fa"):
        self.client = OpenAI(api_key=os.getenv("API_KEY"),
                             base_url="https://api.avalai.ir/v1")
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.chunk_size = 3000  # characters per chunk, adjust based on your needs
        
    def ocr_pdf(self, pdf_path: str, dpi: int = 300) -> List[str]:
        """Extract text from PDF pages using OCR."""
        print(f"Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=dpi)
        
        pages_text = []
        for i, image in enumerate(images):
            print(f"OCR processing page {i+1}/{len(images)}...")
            text = pytesseract.image_to_string(image, lang='eng')  # adjust lang as needed
            pages_text.append(text)
        
        return pages_text
    
    def is_incomplete_sentence(self, text: str) -> bool:
        """Check if text ends with an incomplete sentence."""
        text = text.rstrip()
        if not text:
            return False
        
        last_char = text[-1]
        
        # Complete sentence endings
        if last_char in '.!?。！？':
            return False
        
        # Incomplete patterns (just check last character directly)
        if last_char in ',-:;' or last_char.islower():
            return True
        
        return False

    
    def merge_pages(self, pages_text: List[str]) -> str:
        """Merge pages handling incomplete paragraphs at page boundaries."""
        merged_text = ""
        
        for i, page_text in enumerate(pages_text):
            page_text = page_text.strip()
            
            if i == 0:
                merged_text = page_text
            else:
                # Check if previous page ended incompletely
                if self.is_incomplete_sentence(merged_text):
                    # Remove extra whitespace and merge without paragraph break
                    merged_text = merged_text.rstrip() + " " + page_text.lstrip()
                else:
                    # Add paragraph break between pages
                    merged_text += "\n\n" + page_text
        
        return merged_text
    
    def smart_chunk(self, text: str) -> List[str]:
        """Split text into chunks at natural boundaries (paragraphs/sentences)."""
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # If single paragraph is too large, split by sentences
                if len(para) > self.chunk_size:
                    sentences = re.split(r'([.!?。！？]+\s+)', para)
                    temp = ""
                    
                    for j in range(0, len(sentences), 2):
                        sentence = sentences[j]
                        if j + 1 < len(sentences):
                            sentence += sentences[j + 1]
                        
                        if len(temp) + len(sentence) > self.chunk_size:
                            if temp:
                                chunks.append(temp.strip())
                            temp = sentence
                        else:
                            temp += sentence
                    
                    current_chunk = temp
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def translate_chunk(self, chunk: str, chunk_num: int, total_chunks: int) -> str:
        """Translate a single chunk using Claude."""
        print(f"Translating chunk {chunk_num}/{total_chunks}...")
        
        prompt = f"""Translate the following text from {self.source_lang} to {self.target_lang}.
                    Maintain the original formatting, paragraph breaks, and style.
                    Only provide the translation, no explanations.

                    Text to translate:
                    {chunk}"""
        
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    
    def translate_book(self, pdf_path: str, output_path: str):
        """Complete pipeline: OCR -> merge -> chunk -> translate -> save."""
        # Step 1: OCR
        print("Step 1: OCR extraction...")
        pages_text = self.ocr_pdf(pdf_path)
        
        # Step 2: Merge pages handling incomplete paragraphs
        print("\nStep 2: Merging pages...")
        full_text = self.merge_pages(pages_text)
        
        # Save OCR output for reference
        ocr_output_path = output_path.replace('.txt', '_ocr.txt')
        with open(ocr_output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"OCR text saved to: {ocr_output_path}")
        
        # Step 3: Chunk text
        print("\nStep 3: Chunking text...")
        chunks = self.smart_chunk(full_text)
        print(f"Created {len(chunks)} chunks")
        
        # Step 4: Translate chunks
        print("\nStep 4: Translating...")
        translated_chunks = []
        for i, chunk in enumerate(chunks, 1):
            translated = self.translate_chunk(chunk, i, len(chunks))
            translated_chunks.append(translated)
        
        # Step 5: Save translation
        print("\nStep 5: Saving translation...")
        final_translation = "\n\n".join(translated_chunks)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_translation)
        
        print(f"\nTranslation complete! Saved to: {output_path}")
        return final_translation


# Usage example
if __name__ == "__main__":
    # Set your API key
    API_KEY = os.getenv("ANTHROPIC_API_KEY")  # or paste directly
    
    translator = BookTranslator(
        api_key=API_KEY,
        source_lang="English",
        target_lang="Persian"
    )
    
    # Translate your book
    translator.translate_book(
        pdf_path="input_book.pdf",
        output_path="translated_book.txt"
    )
