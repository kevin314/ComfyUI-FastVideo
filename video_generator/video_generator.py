from __future__ import annotations
import os
import glob

from fastvideo import VideoGenerator as FastVideoGenerator
from fastvideo.v1.configs.models import DiTConfig, EncoderConfig, VAEConfig


MAX_RESOLUTION = 16384

class VideoGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inference_args": ("INFERENCE_ARGS",),
                "prompt": ("STRING",
                           {"multiline": True,
                            "default": "A ripe orange tumbles gently from a tree and lands on the head of a lounging capybara, "
                            "who blinks slowly in response. The moment is quietly humorous and oddly serene, framed by "
                            "lush green foliage and dappled sunlight. Mid-shot, warm and whimsical tones."}),
                "output_path": ("STRING", {"default": "/workspace/ComfyUI/outputs_video/"}),
                "num_gpus": ("INT", {"default": 2, "min": 1, "max": 16}),
                "model_path": ("STRING", {"default": "FastVideo/FastHunyuan-diffusers"}),
                "embedded_cfg_scale": ("FLOAT", {"default": 6.0}),
                "sp_size": ("INT", {"default": 2}),
                "tp_size": ("INT", {"default": 2}),
            },
            "optional": {
                "vae_config": ("VAE_CONFIG",),
                "vae_precision": (["fp16", "bf16"], {"default": "fp16"}),
                "vae_tiling": ([True, False], {"default": True}),
                "vae_sp": ([True, False], {"default": False}),

                "text_encoder_config": ("TEXT_ENCODER_CONFIG",),
                "text_encoder_precision": (["fp16", "bf16"], {"default": "fp16"}),

                "dit_config": ("DIT_CONFIG",),
                "precision": (["fp16", "bf16"], {"default": "fp16"}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "launch_inference"
    CATEGORY = "fastvideo"

    generator = None

    def load_output_video(self, output_dir):
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        video_files = []

        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(output_dir, ext)))

        if not video_files:
            print("No video files found in output directory: %s", output_dir)
            return ""

        video_files.sort()
        return video_files[0]

    def launch_inference(
        self,
        prompt,
        output_path,
        num_gpus,
        model_path,
        embedded_cfg_scale,
        sp_size,
        tp_size,
        vae_precision,
        vae_tiling,
        vae_sp,
        text_encoder_precision,
        precision,
        inference_args=None,
        vae_config=None,
        text_encoder_config=None,
        dit_config=None,
    ):        
        # current_env = os.environ.copy()

        # current_env["PYTHONIOENCODING"] = "utf-8"
        # current_env["FASTVIDEO_ATTENTION_BACKEND"] = ""
        # current_env["MODEL_BASE"] = model_path

        dit_config_obj = DiTConfig()
        vae_config_obj = VAEConfig()
        text_encoder_config_obj = EncoderConfig()

        dit_config_obj.update_model_config(dit_config)
        vae_config_obj.update_model_config(vae_config)
        text_encoder_config_obj.update_model_config(text_encoder_config)

        if self.generator is None:
            self.generator = FastVideoGenerator.from_pretrained(
                model_path=model_path,
                num_gpus=num_gpus,
                tp_size=tp_size,
                sp_size=sp_size,
                embedded_cfg_scale=embedded_cfg_scale,
                vae_precision=vae_precision,
                vae_tiling=vae_tiling,
                vae_sp=vae_sp,
                text_encoder_precision=text_encoder_precision,
                precision=precision,
                dit_config=dit_config_obj,
                vae_config=vae_config_obj,
                text_encoder_config=text_encoder_config_obj
            )

        self.generator.generate_video(
            prompt=prompt,
            output_path=output_path,
            num_inference_steps=inference_args["num_inference_steps"],
            num_frames=inference_args["num_frames"],
            height=inference_args["height"],
            width=inference_args["width"],
            guidance_scale=inference_args["guidance_scale"],
            seed=inference_args["seed"]
        )

        output_path = os.path.join(output_path, f"{prompt[:100]}.mp4")
        return(output_path,)
