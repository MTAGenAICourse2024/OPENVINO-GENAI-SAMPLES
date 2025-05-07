#!/usr/bin/env python3
"""
Setup script for Speech-to-Text module
This script installs all dependencies and checks for OpenVINO installation
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_step(message):
    """Print a formatted step message"""
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)


def run_command(command, cwd=None):
    """Run a shell command and return if it succeeded"""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        print(f"Error: {result.stderr}")
        return False, result.stderr
    
    return True, result.stdout


def check_python_version():
    """Check if Python version is suitable"""
    print_step("Checking Python version")
    
    major, minor = sys.version_info.major, sys.version_info.minor
    print(f"Python version: {major}.{minor}")
    
    if major != 3 or minor < 8:
        print("Warning: This module is tested with Python 3.8+")
        return False
    
    return True


def check_pip():
    """Check if pip is installed and get its version"""
    print_step("Checking pip installation")
    
    success, output = run_command([sys.executable, "-m", "pip", "--version"])
    if not success:
        print("pip is not installed or not working correctly")
        return False
    
    print(f"pip is installed: {output}")
    return True


def install_requirements():
    """Install all required packages"""
    print_step("Installing required packages")
    
    requirements = [
        "numpy>=1.20.0",
        "streamlit>=1.20.0",
        "sounddevice>=0.4.5",
        "scipy>=1.7.0",
        "python-dotenv>=0.19.0",
        "matplotlib>=3.5.0",
        "streamlit-webrtc>=0.44.0",
        "openvino>=2023.0.0",  # Base OpenVINO dependency
    ]
    
    # Install core requirements
    for req in requirements:
        print(f"Installing {req}")
        success, _ = run_command([sys.executable, "-m", "pip", "install", req])
        if not success:
            print(f"Failed to install {req}")
            return False
    
    # Write requirements to a file for future reference
    with open("requirements_stt.txt", "w") as f:
        f.write("\n".join(requirements))
        f.write("\n# Additional requirements may be needed depending on your system\n")
    
    return True


def check_openvino():
    """Check if OpenVINO is installed and working"""
    print_step("Checking OpenVINO installation")
    
    try:
        import openvino
        print(f"OpenVINO is installed: version {openvino.__version__}")
        return True
    except ImportError:
        print("OpenVINO is not installed")
        return False


def check_openvino_genai():
    """Check if openvino_genai is installed"""
    print_step("Checking OpenVINO GenAI installation")
    
    try:
        import openvino_genai
        print(f"OpenVINO GenAI is installed")
        return True
    except ImportError:
        print("OpenVINO GenAI is not installed")
        
        # Try to install it
        success, _ = run_command([sys.executable, "-m", "pip", "install", "openvino-genai"])
        if not success:
            print("Failed to install openvino-genai")
            return False
        
        try:
            import openvino_genai
            print(f"OpenVINO GenAI has been installed")
            return True
        except ImportError:
            print("OpenVINO GenAI could not be installed")
            return False


def check_audio_device():
    """Check if audio devices are available"""
    print_step("Checking audio devices")
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"Found {len(devices)} audio devices:")
        for i, device in enumerate(devices):
            print(f"  {i}: {device['name']} (inputs: {device['max_input_channels']}, outputs: {device['max_output_channels']})")
        
        if not any(device['max_input_channels'] > 0 for device in devices):
            print("Warning: No input devices found")
            return False
        
        return True
    except Exception as e:
        print(f"Error accessing audio devices: {e}")
        return False


def download_whisper_model():
    """Download the OpenVINO Whisper model"""
    print_step("Checking for Whisper model")
    
    model_dir = Path("whisper-base")
    
    if model_dir.exists() and list(model_dir.glob("*.xml")):
        print(f"Whisper model already exists at: {model_dir.absolute()}")
        return True
    
    # Model doesn't exist, download it
    print("Whisper model not found. Downloading...")
    
    try:
        from openvino.tools import download_model
        download_model("whisper-base", output_dir=str(model_dir))
        print(f"Successfully downloaded model to: {model_dir.absolute()}")
        return True
    except Exception as e:
        print(f"Failed to download Whisper model: {e}")
        print("Please download the model manually from https://huggingface.co/openai/whisper-base")
        return False


def setup_streamlit_config():
    """Set up Streamlit configuration"""
    print_step("Setting up Streamlit configuration")
    
    # Create .streamlit directory if it doesn't exist
    config_dir = Path.home() / ".streamlit"
    config_dir.mkdir(exist_ok=True)
    
    # Create or update config.toml
    config_path = config_dir / "config.toml"
    
    config_content = """
[theme]
primaryColor = "#3498db"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
enableCORS = false
enableXsrfProtection = false
fileWatcherType = "none"  # Prevent torch._classes issue

[browser]
serverAddress = "localhost"
gatherUsageStats = false
"""
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"Streamlit configuration written to: {config_path}")
    return True


def create_launcher():
    """Create a launcher script"""
    print_step("Creating launcher script")
    
    launcher_script = """#!/usr/bin/env python3
'''
Launch script for STT app
'''
import os
# Set environment variables to prevent PyTorch/Streamlit issues
os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"

import streamlit.web.cli as stcli
import sys

if __name__ == "__main__":
    # Run the Streamlit app
    sys.argv = ["streamlit", "run", "04_microphone_to_text.py"]
    sys.exit(stcli.main())
"""
    
    with open("launch_stt.py", "w") as f:
        f.write(launcher_script)
    
    # Make it executable on Unix-like systems
    if platform.system() != "Windows":
        os.chmod("launch_stt.py", 0o755)
    
    print("Launcher script created: launch_stt.py")
    print("To run the app: python launch_stt.py")
    return True


def main():
    """Main setup function"""
    print_step("Starting setup for Speech-to-Text module")
    
    # Run all setup steps and track success
    steps = [
        ("Python version", check_python_version),
        ("pip installation", check_pip),
        ("Requirements installation", install_requirements),
        ("OpenVINO installation", check_openvino),
        ("OpenVINO GenAI installation", check_openvino_genai),
        ("Audio device check", check_audio_device),
        ("Whisper model download", download_whisper_model),
        ("Streamlit configuration", setup_streamlit_config),
        ("Launcher creation", create_launcher),
    ]
    
    results = []
    for name, func in steps:
        success = func()
        results.append((name, success))
    
    # Print summary
    print_step("Setup Summary")
    for name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{status}: {name}")
    
    # Check if all steps succeeded
    if all(success for _, success in results):
        print("\nSetup completed successfully! 🎉")
        print("To run the app, execute: python launch_stt.py")
    else:
        print("\nSetup completed with some issues. Please check the output above.")
        print("You may need to fix these issues manually before the app will work correctly.")


if __name__ == "__main__":
    main()