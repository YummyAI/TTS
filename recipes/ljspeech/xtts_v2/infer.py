import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import os
import shutil

checkpoint_dir = "run/ckpt_test"
os.makedirs(checkpoint_dir, exist_ok=True)


# required_files = ["model.pth", "config.json", "vocab.json", "speakers_xtts.pth"]
# files_in_dir = os.listdir(checkpoint_dir)
# 

config_file = os.path.join(checkpoint_dir, "config.json")

print("Loading model...")
config = XttsConfig()
config.load_json(config_file)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir)
model.cuda()

supported_languages = config.languages
if not "vi" in supported_languages:
    supported_languages.append("vi")


print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path="ref_en_wav.wav",gpt_cond_len=30,
                gpt_cond_chunk_len=4,
                max_ref_length=60)

print("Inference...")
out = model.inference(
    "hãy so sánh mình với người nghèo chúng ta sẽ hiểu rằng biết đủ chính là hạnh phúc.",
    "vi",
    gpt_cond_latent,
    speaker_embedding,
    repetition_penalty =5.0,
    temperature=0.75)
torchaudio.save("xtts_vi_test.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)