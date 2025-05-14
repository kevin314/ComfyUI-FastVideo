from __future__ import annotations
import os
import glob

from fastvideo import VideoGenerator as FastVideoGenerator, PipelineConfig
from fastvideo.v1.configs.models import DiTConfig, VAEConfig
from fastvideo.v1.configs.models.encoders import TextEncoderConfig


MAX_RESOLUTION = 16384

def update_config_from_args(config, args_dict):
    """
    Update configuration object from arguments dictionary.
    
    Args:
        config: The configuration object to update
        args_dict: Dictionary containing arguments
    """
    for key, value in args_dict.items():
        if hasattr(config, key) and value is not None:
            if key == "text_encoder_precisions" and isinstance(value, list):
                setattr(config, key, tuple(value))
            else:
                setattr(config, key, value)

class VideoGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING",
                           {"multiline": True,
                            "default": "A ripe orange tumbles gently from a tree and lands on the head of a lounging capybara, "
                            "who blinks slowly in response. The moment is quietly humorous and oddly serene, framed by "
                            "lush green foliage and dappled sunlight. Mid-shot, warm and whimsical tones."}),
                "output_path": ("STRING", {"default": "/workspace/ComfyUI/outputs_video/"}),
                "num_gpus": ("INT", {"default": 2, "min": 1, "max": 16}),
                "model_path": ("STRING", {"default": "FastVideo/FastHunyuan-diffusers"})
            },
            "optional": {
                "inference_args": ("INFERENCE_ARGS",),
                "embedded_cfg_scale": ("FLOAT", {"default": 6.0}),
                "sp_size": ("INT", {"default": 2}),
                "tp_size": ("INT", {"default": 2}),

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
        # Load pipeline config from model path
        pipeline_config = PipelineConfig.from_pretrained(model_path)
        
        # Update configs with provided config dictionaries
        if dit_config is not None:
            update_config_from_args(pipeline_config.dit_config, dit_config)
        
        if vae_config is not None:
            update_config_from_args(pipeline_config.vae_config, vae_config)
            
        if text_encoder_config is not None:
            update_config_from_args(pipeline_config.text_encoder_configs, text_encoder_config)
        
        # Update top-level pipeline config with remaining arguments
        raw_pipeline_args = {}
        if embedded_cfg_scale is not None:
            raw_pipeline_args['embedded_cfg_scale'] = embedded_cfg_scale
        if precision is not None:
            raw_pipeline_args['precision'] = precision
        if vae_precision is not None:
            raw_pipeline_args['vae_precision'] = vae_precision
        if vae_tiling is not None:
            raw_pipeline_args['vae_tiling'] = vae_tiling
        if vae_sp is not None:
            raw_pipeline_args['vae_sp'] = vae_sp
        if text_encoder_precision is not None:
            raw_pipeline_args['text_encoder_precision'] = text_encoder_precision

        # Filter out any value explicitly set to -99999 (auto values)
        pipeline_args = {k: v for k, v in raw_pipeline_args.items() if v != -99999}

        update_config_from_args(pipeline_config, pipeline_args)

        generation_args = {}
        if num_gpus is not None:
            generation_args['num_gpus'] = num_gpus
        if tp_size is not None:
            generation_args['tp_size'] = tp_size
        if sp_size is not None:
            generation_args['sp_size'] = sp_size
            

        # Initialize generator if not already done
        if self.generator is None:
            print('generation_args', generation_args)
            print('pipeline_config', pipeline_config)
            self.generator = FastVideoGenerator.from_pretrained(
                model_path=model_path,
                **generation_args,
                pipeline_config=pipeline_config
            )

        print('inference_args', inference_args)
        # Generate the video
        self.generator.generate_video(
            prompt=prompt,
            output_path=output_path,
            **inference_args
        )

        output_path = os.path.join(output_path, f"{prompt[:100]}.mp4")
        return(output_path,)
