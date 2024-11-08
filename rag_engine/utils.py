import re
import os
import base64
import fitz
import torch
from io import BytesIO
from PIL import Image
import requests


PAGE_PROMT_TEMP = """
give the name of the main headlines of this page

btw for more context this is {}.

Here is the text from the target page:
{}.

Answer in one sentence only! directly give output
"""

CHUNK_PROMT_TEMP = """
describe what this chunk of text is about giving a context about thw whole page
Here’s the page description: {}. 

Now, here’s the specific chunk: {}


it should be short sentence, maximum 15 words. dont repeat the chunk text  
directly give output: 

"""

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def filter_encoded_text(text):
    # Remove long sequences that look like base64 encoding
    # Adjust length (100 here) based on your data
    filtered_text = re.sub(r'\b[A-Za-z0-9+/]{100,}={0,2}\b', '', text)
    return filtered_text

def page2chunks(page_text, chunk_size=1000):
    text_blocks = page_text.split('\n\n')
    final_chunks = adjust_chunks(text_blocks, chunk_size)
    return final_chunks
             
def adjust_chunks(chunks, target_size):
    adjusted_chunks = []
    buffer = ""

    for chunk in chunks:
        if len(buffer) == 0:
            buffer = chunk
        else:
            buffer += "\n\n" + chunk

        # If buffer exceeds the target size, split and adjust
        while len(buffer) > target_size:
            # Split the buffer around the target size boundary
            split_point = buffer[:target_size].rfind(" ")  # Split at the last space before target_size
            if split_point == -1:
                split_point = target_size  # If no space found, hard split at target_size
            
            # Add the split portion to the adjusted chunks
            adjusted_chunks.append(buffer[:split_point])
            buffer = buffer[split_point:].lstrip()  # Start a new buffer with the remainder

        # If buffer is close to target size, add it to adjusted chunks
        if len(buffer) >= target_size * 0.8 and len(buffer) <= target_size * 1.2:
            adjusted_chunks.append(buffer)
            buffer = ""  # Reset buffer

    # Add any remaining text in the buffer as a final chunk
    if buffer:
        adjusted_chunks.append(buffer)

    return adjusted_chunks


def process_text_blocks(text_blocks, char_count_threshold=500):
    """Group text blocks based on a character count threshold."""
    current_group = []
    grouped_blocks = []
    current_char_count = 0

    for block in text_blocks:
        if block[-1] == 0:  # Check if the block is of text type
            block_text = block[4]
            block_char_count = len(block_text)

            if current_char_count + block_char_count <= char_count_threshold:
                current_group.append(block)
                current_char_count += block_char_count
            else:
                if current_group:
                    grouped_content = "\n".join([b[4] for b in current_group])
                    grouped_blocks.append((current_group[0], grouped_content))
                current_group = [block]
                current_char_count = block_char_count

    # Append the last group
    if current_group:
        grouped_content = "\n".join([b[4] for b in current_group])
        grouped_blocks.append((current_group[0], grouped_content))

    return grouped_blocks


# from llama_index.legacy.llms.generic_utils import (
#     messages_to_prompt as generic_messages_to_prompt,
# )


def hf_autotokenizer_to_chat_formatter( pretrained_model_name_or_path):
    # https://huggingface.co/docs/transformers/main/chat_templating
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer_config.json
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)  # type: ignore

    def format_autotokenizer(
        messages,
        **kwargs,
    ):
        tokenizer.use_default_system_prompt = False  # type: ignore
        
        messages_dict = [
                {"role": message.role.value, "content": message.content}
                for message in messages
            ]
        prompt: str = tokenizer.apply_chat_template(messages_dict, tokenize=False)  # type: ignore
        return prompt
        # assert isinstance(prompt, str)
        # # Return formatted prompt and eos token by default
        # return ChatFormatterResponse(
        #     prompt=prompt, stop=tokenizer.eos_token, added_special=True
        # )
        # return generic_messages_to_prompt(prompt)

    return format_autotokenizer