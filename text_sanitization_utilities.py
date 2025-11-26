import re
import os

def enhanced_gtts_speech(text, language="English", output_path=None, slow=False):
    """
    Enhanced gTTS speech generation with improved parameters and text preprocessing.
    
    Args:
        text (str): Text to convert to speech
        language (str): Language for TTS
        output_path (str): Path to save audio file
        slow (bool): Whether to use slow speech rate
        
    Returns:
        str: Path to generated audio file
    """
    try:
        from gtts import gTTS
        import tempfile
        
        # Language codes for better pronunciation
        language_codes = {
            "English": "en",
            "Hebrew": "he",  # Use 'he' instead of 'iw' for better support
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
            "Japanese": "ja",
            "Chinese": "zh"
        }
        
        lang_code = language_codes.get(language, "en")
        
        # Preprocess text for better intonation
        clean_text = improve_tts_intonation(text)
        
        # Create temporary file if no output path specified
        if not output_path:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            output_path = temp_file.name
            temp_file.close()
        
        # Enhanced gTTS parameters
        tts = gTTS(
            text=clean_text,
            lang=lang_code,
            slow=slow,
            lang_check=True,  # Enable language checking
            pre_processor_funcs=[
                # Add custom preprocessing functions if needed
            ]
        )
        
        # Save with error handling
        tts.save(output_path)
        
        return output_path
        
    except Exception as e:
        print(f"Enhanced gTTS error: {e}")
        return None



def sanitize_filename_bilingual(bilingual):
# For the filename, create a safe version of the prompt
    safe_filename = "".join([c for c in bilingual[:30] if c.isalnum() or c.isspace()]).strip()
    safe_filename = safe_filename.replace(" ", "_")
    
   
    #filename = f"generated_images/generated_image_{safe_filename}_{timestamp}.png"
    return safe_filename

def sanitize_filename(prompt, max_length=15):
    """
    Convert a prompt into a valid filename by:
        1. Removing spaces and special characters
        2. Limiting the length to max_length characters
        
    Args:
        prompt (str): The original prompt text
        max_length (int): Maximum length for the filename (default: 15)
            
    Returns:
        str: A sanitized filename
    """

    if not prompt:
        return "unnamed"
    # Convert to lowercase for consistency
    filename = prompt.lower()
    


    # Replace spaces with underscores
    filename = filename.replace(' ', '_')

        # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "")

    # Remove any non-alphanumeric characters except underscores
    filename = re.sub(r'[^a-z0-9_]', '', filename)

    # Remove any remaining problematic characters
    filename = re.sub(r'[^\w\-_.]', '', filename)
        
    # Limit to max_length characters
    if len(filename) > max_length:
        filename = filename[:max_length]
            
    # Make sure we don't end with an underscore
    if filename.endswith('_'):
        filename = filename[:-1]
            
    # If filename is empty after sanitizing, use a default name
    if not filename:
        filename = "image"
        
    return filename




def sanitize_story(story):
    """Sanitize the story by removing unwanted text and limiting length"""
    # Remove code blocks (text between ```)
    story = re.sub(r'```[\s\S]*?```', '', story)
    # Remove statements that start with "Fun 10-word", "Seuss", or "10-word"
    story = re.sub(r'Fun 10-word[^.]*\.?', '', story)
    story = re.sub(r'Fun 10-Word[^.]*\.?', '', story)
    story = re.sub(r'Seuss[^.]*\.?', '', story)
    story = re.sub(r'10-Word[^.]*\.?', '', story)
    story = re.sub(r'10-word[^.]*\.?', '', story)
    # Remove sections starting with "Sure"
    story = re.sub(r'Sure[^.]*\.?', '', story)
    # Remove sections starting with "**Python Function:**"
    story = re.sub(r'\*\*Python Function:\*\*[\s\S]*?(?=\n\n|\Z)', '', story)
    # Remove code blocks (text between ```)
    story = re.sub(r'```[\s\S]*?```', '', story)
    # Remove statements that start with "Fun 10-word", "Seuss", or "10-word"
    story = re.sub(r'Fun 10-word[^.]*\.?', '', story)
    story = re.sub(r'Fun 10-Word[^.]*\.?', '', story)
    story = re.sub(r'Seuss[^.]*\.?', '', story)
    story = re.sub(r'10-Word[^.]*\.?', '', story)
    story = re.sub(r'10-word[^.]*\.?', '', story)
    # Remove sections starting with "Sure"
    story = re.sub(r'Sure[^.]*\.?', '', story)
                                
    # Remove extra whitespace
    story = re.sub(r'\s+', ' ', story).strip()
                                
    return story



def sanitize_fun_saying(fun_saying):
    """Sanitize the fun saying by removing unwanted text and limiting length"""
    
    # Remove code blocks (text between ```)
    fun_saying = re.sub(r'```[\s\S]*?```', '', fun_saying)
    # Remove statements that start with "Fun 10-word", "Seuss", or "10-word"
    fun_saying = re.sub(r'Fun 10-word[^.]*\.?', '', fun_saying)
    fun_saying = re.sub(r'Fun 10-Word[^.]*\.?', '', fun_saying)
    fun_saying = re.sub(r'Seuss[^.]*\.?', '', fun_saying)
    fun_saying = re.sub(r'10-Word[^.]*\.?', '', fun_saying)
    fun_saying = re.sub(r'10-word[^.]*\.?', '', fun_saying)
    # Remove sections starting with "Sure"
    fun_saying = re.sub(r'Sure[^.]*\.?', '', fun_saying)
                                
    # Remove extra whitespace
    fun_saying = re.sub(r'\s+', ' ', fun_saying).strip()
                                
    # Limit to 150 characters
    if len(fun_saying) > 150:
        fun_saying = fun_saying[:147] + "..."
    return fun_saying


def improve_tts_intonation(text):
    """
    Enhance text for better TTS intonation and naturalness.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text optimized for natural TTS pronunciation
    """
    # Add pauses for better pacing
    text = re.sub(r'([.!?])\s+', r'\1... ', text)  # Add pauses after sentences
    text = re.sub(r'([,;:])\s+', r'\1. ', text)    # Add short pauses after commas/semicolons
    
    # Improve number pronunciation
    text = re.sub(r'\b(\d+)%', r'\1 percent', text)
    text = re.sub(r'\$(\d+)', r'\1 dollars', text)
    text = re.sub(r'\b(\d{4})\b', lambda m: f"{m.group(1)[:2]} {m.group(1)[2:]}" if len(m.group(1)) == 4 else m.group(1), text)
    
    # Expand common abbreviations for clarity
    expansions = {
        r'\bDr\.\s': 'Doctor ',
        r'\bMr\.\s': 'Mister ',
        r'\bMrs\.\s': 'Missus ',
        r'\bMs\.\s': 'Miss ',
        r'\bProf\.\s': 'Professor ',
        r'\bvs\.\s': 'versus ',
        r'\bUS\b': 'United States',
        r'\bUK\b': 'United Kingdom',
        r'\bAI\b': 'artificial intelligence',
        r'\bML\b': 'machine learning',
        r'\bAPI\b': 'A P I',
        r'\bCPU\b': 'C P U',
        r'\bGPU\b': 'G P U',
        r'\bHTML\b': 'H T M L',
        r'\bCSS\b': 'C S S',
        r'\bJS\b': 'JavaScript',
        r'\bURL\b': 'U R L',
        r'\bHTTP\b': 'H T T P',
    }
    
    for pattern, replacement in expansions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Add emphasis for important words
    text = re.sub(r'\b(important|critical|essential|key|significant)\b', r'*\1*', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(first|second|third|finally|lastly)\b', r'*\1*', text, flags=re.IGNORECASE)
    
    # Improve punctuation for better pacing
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1.. \2', text)  # Longer pause between sentences
    text = re.sub(r':\s*([A-Z])', r': \1', text)           # Pause after colons
    
    # Handle lists better
    text = re.sub(r'\n\s*[-•*]\s*', '. Next point: ', text)
    text = re.sub(r'\b(\d+)\.\s+', r'Point \1: ', text)
    
    # Clean up multiple dots/pauses
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def sanitize_srt_for_tts(srt_text):
    """
    Sanitize SRT content to make it more suitable for text-to-speech engines.
    
    Args:
        srt_text (str): Raw SRT content including timestamps and text
        
    Returns:
        str: Cleaned text optimized for TTS engines
    """
    # Extract only the text content from SRT (ignoring timestamps and numbering)
    segments = []
    lines = srt_text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
            
        # Skip SRT numbering (just digits)
        if line.isdigit():
            i += 1
            continue
            
        # Skip timestamp lines (containing "-->")
        if "-->" in line:
            i += 1
            continue
            
        # This should be text content - add to segments
        segments.append(line)
        i += 1
    
    # Join all text segments
    text_content = " ".join(segments)
    
    # Remove quotes (both single and double)
    text_content = text_content.replace('"', '').replace("'", '')
    
    # Remove bracketed content like [Music] or [Applause]
    text_content = re.sub(r'\[.*?\]', '', text_content)
    
    # Remove parenthesized content like (whispering) or (laughs)
    text_content = re.sub(r'\(.*?\)', '', text_content)
    
    # Remove HTML/XML tags
    text_content = re.sub(r'<.*?>', '', text_content)
    
    # Remove speaker identifications like "John:" or "PRESENTER:"
    text_content = re.sub(r'^\s*[A-Z][A-Za-z\s]*:', '', text_content)
    
    # Replace multiple spaces with a single space
    text_content = re.sub(r'\s+', ' ', text_content)
    
    # Replace special characters that might cause TTS issues
    text_content = text_content.replace('&', ' and ')
    text_content = text_content.replace('/', ' or ')
    text_content = text_content.replace('**', '')  # Remove markdown bold
    text_content = text_content.replace('*', '')   # Remove markdown italic
    
    # Remove hashtags, section markers
    text_content = re.sub(r'#+ ', '', text_content)
    
    # Convert some common abbreviations
    text_content = re.sub(r'\bi\.e\.\s', 'that is ', text_content, flags=re.IGNORECASE)
    text_content = re.sub(r'\be\.g\.\s', 'for example ', text_content, flags=re.IGNORECASE)
    text_content = re.sub(r'\betc\.\s', 'etcetera ', text_content, flags=re.IGNORECASE)
    
    # Clean up any remaining issues
    text_content = text_content.strip()
    
    # Apply intonation improvements
    text_content = improve_tts_intonation(text_content)
    
    return text_content

