from pathlib import Path

import soundfile as sf
from dots_tts.runtime import DotsTtsRuntime


# 1. Load the model
print("Loading dots.tts model...")

runtime = DotsTtsRuntime.from_pretrained(
    "dots-studio/dots.tts-soar",
    precision="float32",   # important: your Mac does not have CUDA
    optimize=False,
    max_generate_length=80,
)

print("Model loaded!")


# 2. Generate speech
print("Generating speech...")

result = runtime.generate(
    text="Hello, this is my first dots TTS experiment.",
    prompt_audio_path="my_files/abhi1.wav",
    prompt_text=(
        "Hey, Adam here. Let's create something that feels real, "
        "sounds human, and connects every time."
    ),
    num_steps=10,
    guidance_scale=1.2,
)


# 3. Convert the output tensor into something we can save
audio = result["audio"].float().cpu().squeeze().numpy()


# 4. Save it
output_path = Path("audio/output_nfe10.wav")

sf.write(
    output_path,
    audio,
    result["sample_rate"],
)

print(f"Done! Saved audio to: {output_path}")
