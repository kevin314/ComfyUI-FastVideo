from .video_generator import VideoGenerator
from .inference_args import InferenceArgs
from .vae_config import VAEConfig
from .text_encoder_config import TextEncoderConfig
from .dit_config import DITConfig

NODE_CLASS_MAPPINGS = {
    "VideoGenerator": VideoGenerator,
    "InferenceArgs": InferenceArgs,
    "VAEConfig": VAEConfig,
    "TextEncoderConfig": TextEncoderConfig,
    "DITConfig": DITConfig
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoGenerator": "Video Generator",
    "InferenceArgs": "Inference Args",
    "VAEConfig": "VAE Config",
    "TextEncoderConfig": "Text Encoder Config",
    "DITConfig": "DIT Config"
}