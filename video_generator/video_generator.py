from __future__ import annotations
import os
import glob

from fastvideo import VideoGenerator as FastVideoGenerator


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
                "master_port": ("INT", {"default": 29503}),
                "model_path": ("STRING", {"default": "FastHunyuan-diffusers"}),
                "embedded_cfg_scale": ("FLOAT", {"default": 6.0}),
                "sp_size": ("INT", {"default": 2}),
                "tp_size": ("INT", {"default": 2}),
            },
            "optional": {
                "vae_config": ("VAE_CONFIG",),
                "text_encoder_config": ("TEXT_ENCODER_CONFIG",),
                "dit_config": ("DIT_CONFIG",),
            }
        }

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
        master_port,
        model_path,
        embedded_cfg_scale,
        sp_size,
        tp_size,
        inference_args=None,
        vae_config=None,
        text_encoder_config=None,
        dit_config=None,
    ):        
        current_env = os.environ.copy()

        current_env["PYTHONIOENCODING"] = "utf-8"
        current_env["FASTVIDEO_ATTENTION_BACKEND"] = ""
        current_env["MODEL_BASE"] = model_path

        if self.generator is None:
            self.generator = FastVideoGenerator.from_pretrained(
                model_path="FastVideo/FastHunyuan-diffusers",
                num_gpus=num_gpus,
                tp_size=tp_size,
                sp_size=sp_size,
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
