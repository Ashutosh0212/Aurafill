"""Audio to text conversion with GPT-3.5 refinement.

This script converts audio files to text using OpenAI's Whisper model
and then refines the text using GPT-3.5 to make it more coherent and relevant.
"""

import os
from pathlib import Path
import tempfile
from typing import Optional
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AudioProcessor:
    """Handles audio processing and text refinement."""

    def __init__(self):
        """Initialize the AudioProcessor with OpenAI client."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio file to text using OpenAI's Whisper model.

        Args:
            audio_file_path: Path to the audio file.

        Returns:
            Transcribed text from the audio.
        """
        with open(audio_file_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text

    def refine_text(self, text: str) -> str:
        """Refine the transcribed text using GPT-3.5.

        Args:
            text: Raw transcribed text.

        Returns:
            Refined and more coherent text.
        """
        prompt = f"""Please refine the following transcribed text to make it more coherent, 
clear, and properly formatted. Fix any grammar issues and organize it into proper paragraphs:

{text}

Refined text:"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content

def save_text(text: str, output_path: str):
    """Save the refined text to a file.

    Args:
        text: Text to save.
        output_path: Path where to save the text file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def main():
    """Main Streamlit interface for audio to text conversion."""
    st.set_page_config(page_title="Audio to Text Converter", layout="wide")
    st.title("Audio to Text Converter")

    # Initialize processor
    processor = AudioProcessor()

    # File upload
    audio_file = st.file_uploader(
        "Upload Audio File",
        type=["mp3", "wav", "m4a", "ogg"],
        help="Upload an audio file to convert to text"
    )

    if audio_file:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name

        try:
            with st.spinner("Transcribing audio..."):
                # Transcribe audio
                raw_text = processor.transcribe_audio(tmp_path)
                st.subheader("Raw Transcription")
                st.text_area("Raw text:", value=raw_text, height=200, disabled=True)

            with st.spinner("Refining text..."):
                # Refine text
                refined_text = processor.refine_text(raw_text)
                st.subheader("Refined Text")
                st.text_area("Refined text:", value=refined_text, height=300, disabled=True)

            # Save options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Raw Text"):
                    save_text(raw_text, "raw_transcription.txt")
                    st.success("Raw text saved to 'raw_transcription.txt'")
            
            with col2:
                if st.button("Save Refined Text"):
                    save_text(refined_text, "refined_transcription.txt")
                    st.success("Refined text saved to 'refined_transcription.txt'")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

    # Usage instructions
    with st.expander("Usage Instructions"):
        st.markdown("""
        1. Upload an audio file (supported formats: MP3, WAV, M4A, OGG)
        2. Wait for the transcription process to complete
        3. Review both raw and refined transcriptions
        4. Save either version as needed
        
        Note: The refined text is processed using GPT-3.5 to improve clarity and coherence.
        """)

if __name__ == "__main__":
    main() 