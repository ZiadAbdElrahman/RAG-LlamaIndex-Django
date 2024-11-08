
import os
import fitz
import subprocess
from llama_index.core import Document
from utils import (
    remove_html_tags, filter_encoded_text, adjust_chunks, PAGE_PROMT_TEMP, CHUNK_PROMT_TEMP, page2chunks, process_text_blocks
)

from config import config


def get_pdf_documents(pdf_file):
    """Process a PDF file and extract text, tables, and images."""
    all_pdf_documents = []
    ongoing_tables = {}

    try:
        f = fitz.open(pdf_file, filetype="pdf")
    except Exception as e:
        print(f"Error opening or processing the PDF file: {e}")
        return []

    for i in range(len(f)):
        page = f[i]
        text_blocks = [block for block in page.get_text("blocks", sort=True) 
                       if block[-1] == 0 and not (block[1] < page.rect.height * 0.1 or block[3] > page.rect.height * 0.9)]
        grouped_text_blocks = process_text_blocks(text_blocks)

        for text_block_ctr, (heading_block, content) in enumerate(grouped_text_blocks, 1):
            heading_bbox = fitz.Rect(heading_block[:4])
            bbox = {"x1": heading_block[0], "y1": heading_block[1], "x2": heading_block[2], "x3": heading_block[3]}
            text_doc = Document(
                text=f"{heading_block[4]}\n{content}",
                metadata={
                    **bbox,
                    "type": "text",
                    "page_num": i,
                    "source": f"{pdf_file[:-4]}-page{i}-block{text_block_ctr}"
                },
                id_=f"{pdf_file[:-4]}-page{i}-block{text_block_ctr}"
            )
            all_pdf_documents.append(text_doc)

    f.close()
    return all_pdf_documents

def get_pdf_documents_with_context(pdf_file, llm):
    """Process a PDF file and extract text, tables, and images."""
    all_pdf_documents = []
    ongoing_tables = {}

    try:
        f = fitz.open(pdf_file, filetype="pdf")
    except Exception as e:
        print(f"Error opening or processing the PDF file: {e}")
        return []
    previous_page_description = None
    for i in range(len(f)):
        page = f[i]
        page_text = filter_encoded_text(remove_html_tags(page.get_text("text")))
        page_description = get_page_description(page_text, i+1, llm)
        text_blocks = page2chunks(page_text)
        for chunk_i, text_chunk in enumerate(text_blocks):
            chunk_description = get_chunk_description(text_chunk, page_description, llm)
            text_doc = Document(
                text=f"chunk_description {chunk_description}\n\n{text_chunk}",
                metadata={
                    "type": "text",
                    "page_num": i,
                    "source": f"{pdf_file[:-4]}-page{i}-block{chunk_i}"
                },
                id_=f"{pdf_file[:-4]}-page{i}-block{chunk_i}"
            )
            all_pdf_documents.append(text_doc)
        previous_page_description = page_description
    f.close()
    return all_pdf_documents

def get_page_description(page_text, previous_page_description, llm):
    if previous_page_description is None:
        input_prompt = PAGE_PROMT_TEMP.format('the first page', page_text)
    else:
        input_prompt = PAGE_PROMT_TEMP.format(f'page number: {previous_page_description}', page_text)
        
    page_description = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": "You are an assistant, and yout profession is simplifying text content and summaries it"},
            {
                "role": "user",
                "content": input_prompt
            }
        ]
        )['choices'][0]['message']['content']
    return page_description

def get_chunk_description(chunk_text, page_description, llm):
    chunk_description = llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are an assistant, and yout profession is simplifying text content and summaries it"},
        {
            "role": "user",
            "content": CHUNK_PROMT_TEMP.format(page_description, chunk_text)
        }
    ]
    )['choices'][0]['message']['content']
    return chunk_description

if __name__ == '__main__':
    pass