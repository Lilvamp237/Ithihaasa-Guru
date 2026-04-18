import json
import re

def clean_history_data(text):
    if not text:
        return text
    
    # 1. Strip the "Tick" artifact (Unicode Zero Width Non-Joiner)
    text = text.replace('\u200c', '')

    # 2. Fix Split Diacritics (Systematic OCR errors for Ra-kaara and Yansaya)
    # Joins "ප් රා" -> "ප්‍රා", "ක් ර" -> "ක්‍ර", "ව් යා" -> "ව්‍යා"
    text = re.sub(r'([ක-ෆ])්\s+ර', r'\1්‍ර', text)
    text = re.sub(r'([ක-ෆ])්\s+ය', r'\1්‍ය', text)
    
    # 3. Standardize Footer Artifacts (Repetitive OCR noise in the metadata)
    # Replaces various misreads of "නොමිලේ බෙදාහැරීම සඳහා ය"
    footer_patterns = [
        r'නොංම්ලේ\s+කොදරරැංඊම\s+G3EHI\s+ය',
        r'නොංමලෙ\s+කෙළදරැර\s+පළඳා\s+ය',
        r'නොංමලෙ\'\s+කෙළරැංඊරම\s+BEHI\s+ය'
    ]
    for pattern in footer_patterns:
        text = re.sub(pattern, 'නොමිලේ බෙදාහැරීම සඳහා ය.', text)

    # 4. Contextual Character Swaps (Common systematic misreads)
    replacements = {
        "pුවක්": "පුවක්",          # Chunk 27/28
        "කුයාමාර්ග": "ක්‍රියාමාර්ග",  # Chunk 29
        " COS ": " විට ",           # Chunk 11
        "Bod.": "පවතී.",           # From your sample cleanup for Page 6/8
        "දුක්වේ.": "දැක්වේ.",       # Chunk 5
        "පැවර්මට": "පැවරීමට",      # Chunk 5
        "අතහැර්මට": "අතහැරීමට",     # Chunk 8
        "ප්‍රථමඋටයෙන්": "ප්‍රථමයෙන්", # Chunk 4
        "OLN ": "දොන් ",            # Misread for "Don Juan" - Chunk 7
        "අසවෙදු": "අසවේදු",         # Chunk 9
        "රුපය": "රූපය",            # Common vowel length error
        "\"%/0('": "VOC",           # Misread English - Chunk 36
        "«Bs": "දැකිය",             # Chunk 256
        "ෙස ද": "ලෙස ද",           # Chunk 304
        "සූදුනම්": "සූදානම්",       # Chunk 330
        "කොලරය": "කෝරළය",          # Chunk 335
        "දුමූහ": "දැමූහ",            # Chunk 335
        "මව්බිත්සය": "මව්බිම",      # Chunk 34
        "ජ්‍ය්‍යෝෂ්ඨ": "ජ්‍යෙෂ්ඨ",     # Chunk 63
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 5. List/Bullet Standardization
    # Replaces '76', '7ල', and '7”' artifacts found at the start of lists
    text = re.sub(r'\s76\s', ' • ', text)
    text = re.sub(r'\s7ල\s', ' • ', text)
    text = re.sub(r'\s7”\s', ' • ', text)

    # 6. Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Implementation on your JSON structure
def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data.get('metadata', []):
        entry['text'] = clean_history_data(entry['text'])

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Run the process
process_json('C:\\Github Projects\\Ithihaasa-Guru\\vector_store\\history_meta.json', 'C:\\Github Projects\\Ithihaasa-Guru\\vector_store\\history_meta_cleaned.json')