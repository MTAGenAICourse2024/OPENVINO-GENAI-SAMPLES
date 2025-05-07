import streamlit as st
import os
import numpy as np
import threading
from datetime import datetime
import time
import queue
import sounddevice as sd
import openvino_genai
from typing import Dict, Tuple, Optional, List, Callable


class AudioProcessor:
    """Handles audio capture and processing."""
    
    def __init__(self, sample_rate=16000, block_size=80000):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.audio_queues = {}
        self.terminate_flag = threading.Event()
        
    def create_queue(self, name):
        """Create a new named queue for audio processing."""
        self.audio_queues[name] = queue.Queue()
        return self.audio_queues[name]
        
    def get_queue(self, name):
        """Get an existing queue by name."""
        return self.audio_queues.get(name)
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for the audio stream that distributes audio to all queues."""
        if status:
            print(f"Audio status: {status}")
            
        input_time = datetime.now()
        audio_capture = indata.copy()
        
        # Add the incoming audio data to all processing queues
        for q in self.audio_queues.values():
            q.put((input_time, audio_capture))
    
    def start_stream(self):
        """Start the audio input stream."""
        self.terminate_flag.clear()
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size
        )
        self.stream.start()
        return self.stream
        
    def stop_stream(self):
        """Stop the audio input stream."""
        self.terminate_flag.set()
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        # Clear all queues
        for q in self.audio_queues.values():
            while not q.empty():
                q.get()


class TranscriptionManager:
    """Manages transcription models and processing threads."""
    
    def __init__(self):
        self.transcribers = {}
        self.processing_threads = []
        self.terminate_flag = threading.Event()
        self.result_queue = queue.Queue()
        self.first_transcription_completed = threading.Event()
    
    def initialize_transcriber(self, device_type, model_name="whisper-base"): # "whisper_small_fp16") 
        """Initialize a transcriber for the specified device."""
        try:
            transcriber = openvino_genai.WhisperPipeline(model_name, device_type)
            self.transcribers[device_type] = transcriber
            print(f"Initialized {device_type} transcriber with model {model_name}")
            return True
        except Exception as e:
            print(f"Failed to initialize {device_type} transcriber: {e}")
            return False
    
    def process_audio(self, device_type, audio_queue):
        """Process audio from queue and run transcription for specified device."""
        counter = 0
        print(f"Starting {device_type} processing thread")
        
        while not self.terminate_flag.is_set():
            try:
                item = audio_queue.get(timeout=1)
                if item is None:
                    break
                input_time, indata = item
            except queue.Empty:
                continue
                
            try:
                # Preprocess audio
                audio_data = np.squeeze(indata).astype(np.float32)
                
                # Run transcription
                transcriber = self.transcribers.get(device_type)
                if transcriber is None:
                    continue
                    
                result = transcriber.generate(audio_data)
                if not isinstance(result, str):
                    result = str(result)
                    
                # Process result
                counter += 1
                output_time = datetime.now()
                latency = (output_time - input_time).total_seconds()
                
                self.result_queue.put((
                    device_type.lower(),
                    counter,
                    result,
                    latency
                ))
                
                self.first_transcription_completed.set()
                print(f"{device_type}: {result}")
                
            except Exception as e:
                print(f"Error processing {device_type} audio: {e}")
                
        print(f"{device_type} thread exiting...")
    
    def start_processing(self, audio_queues):
        """Start processing threads for all initialized transcribers."""
        self.terminate_flag.clear()
        self.processing_threads = []
        
        for device_type, transcriber in self.transcribers.items():
            if device_type in audio_queues:
                thread = threading.Thread(
                    target=self.process_audio,
                    args=(device_type, audio_queues[device_type])
                )
                thread.start()
                self.processing_threads.append(thread)
        
        return len(self.processing_threads)
    
    def stop_processing(self):
        """Stop all processing threads."""
        self.terminate_flag.set()
        
        for thread in self.processing_threads:
            thread.join(timeout=2)
        
        self.processing_threads = []


class STTStreamUI:
    """Streamlit UI for speech-to-text streaming."""
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.transcription_manager = TranscriptionManager()
        
        # Initialize session state
        if "transcription_states" not in st.session_state:
            st.session_state.transcription_states = {}
    
    def setup_ui(self):
        """Set up the Streamlit UI elements."""
        st.title("Real-time Speech-to-Text")
        
        with st.sidebar:
            device_options = ["CPU", "NPU", "BOTH"]
            selected_devices = st.multiselect(
                "Select Devices", 
                device_options,
                default=["CPU"]
            )
            
            language_code = st.selectbox(
                "Language", 
                ["en", "fr", "de", "es", "it", "ja", "zh"],
                index=0
            )
            
            translate = st.checkbox("Translate to English", value=False)
            
            start_button = st.button("Start Transcription")
            stop_button = st.button("Stop Transcription")
        
        return {
            "selected_devices": selected_devices,
            "language_code": language_code,
            "translate": translate,
            "start_button": start_button,
            "stop_button": stop_button
        }
    
    def setup_transcription_ui(self, devices):
        """Create the transcription display UI."""
        columns = []
        placeholders = {}
        
        if len(devices) == 1:
            placeholders[devices[0]] = st.empty()
        else:
            cols = st.columns(len(devices))
            for i, device in enumerate(devices):
                placeholders[device] = cols[i].empty()
                
        return placeholders
    
    def update_transcription_ui(self, placeholders):
        """Update the transcription UI with new results."""
        result_queue = self.transcription_manager.result_queue
        
        while not result_queue.empty():
            device, counter, text, latency = result_queue.get()
            
            # Initialize device state if needed
            if device not in st.session_state.transcription_states:
                st.session_state.transcription_states[device] = ""
            
            # Update the transcription text
            device_text = st.session_state.transcription_states[device]
            device_text += f"{counter}: {text}\n"
            st.session_state.transcription_states[device] = device_text
            
            # Update the UI
            if device.upper() in placeholders:
                with placeholders[device.upper()]:
                    st.text_area(
                        f"{device.upper()} Transcription", 
                        value=device_text,
                        height=200
                    )
        
        # Auto-scroll text areas
        self._auto_scroll_text_areas()
    
    def _auto_scroll_text_areas(self):
        """Helper to auto-scroll text areas to bottom."""
        js = """
        <script>
            function scroll(dummy_var_to_force_repeat_execution){
                var textAreas = parent.document.querySelectorAll('.stTextArea textarea');
                for (let index = 0; index < textAreas.length; index++) {
                    textAreas[index].scrollTop = textAreas[index].scrollHeight;
                }
            }
            scroll(%s)
        </script>
        """ % time.time()  # Use current time to force re-execution
        st.components.v1.html(js)
    
    def start_transcription(self, selected_devices, language_code, translate):
        """Start the transcription process."""
        try:
            # Clear previous state
            st.session_state.transcription_states = {}
            
            # Initialize audio queues
            audio_queues = {}
            for device in selected_devices:
                if device != "BOTH":
                    audio_queues[device] = self.audio_processor.create_queue(device)
            
            if "BOTH" in selected_devices:
                audio_queues["CPU"] = self.audio_processor.create_queue("CPU")
                audio_queues["NPU"] = self.audio_processor.create_queue("NPU")
                selected_devices = ["CPU", "NPU"]
            
            # Initialize transcribers
            for device in audio_queues.keys():
                success = self.transcription_manager.initialize_transcriber(device)
                if not success:
                    st.error(f"Failed to initialize {device} transcriber")
                    return False
            
            # Create UI placeholders
            placeholders = self.setup_transcription_ui(list(audio_queues.keys()))
            
            # Start processing
            self.transcription_manager.start_processing(audio_queues)
            
            # Start audio stream
            self.audio_processor.start_stream()
            
            # Update UI until stopped
            while not self.transcription_manager.terminate_flag.is_set():
                self.update_transcription_ui(placeholders)
                time.sleep(0.1)
            
            return True
        
        except Exception as e:
            st.error(f"Error starting transcription: {e}")
            return False
        
    def stop_transcription(self):
        """Stop the transcription process."""
        self.transcription_manager.stop_processing()
        self.audio_processor.stop_stream()


def run_stt_app():
    """Main function to run the STT app."""
    ui = STTStreamUI()
    ui_controls = ui.setup_ui()
    
    if ui_controls["start_button"]:
        ui.start_transcription(
            ui_controls["selected_devices"],
            ui_controls["language_code"],
            ui_controls["translate"]
        )
    
    if ui_controls["stop_button"]:
        ui.stop_transcription()


# Run the app when this file is executed directly
if __name__ == "__main__":
    run_stt_app()
