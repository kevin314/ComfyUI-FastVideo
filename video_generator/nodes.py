from .video_generator import VideoGenerator
from .inference_args import InferenceArgs
from .vae_config import VAEConfig
from .text_encoder_config import TextEncoderConfig
from .dit_config import DITConfig
from .load_image import LoadImagePath

NODE_CLASS_MAPPINGS = {
    "VideoGenerator": VideoGenerator,
    "InferenceArgs": InferenceArgs,
    "VAEConfig": VAEConfig,
    "TextEncoderConfig": TextEncoderConfig,
    "DITConfig": DITConfig,
    "LoadImagePath": LoadImagePath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoGenerator": "Video Generator",
    "InferenceArgs": "Inference Args",
    "VAEConfig": "VAE Config",
    "TextEncoderConfig": "Text Encoder Config",
    "DITConfig": "DIT Config",
    "LoadImagePath": "Load Image Path"
}