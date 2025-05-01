class InferenceArgs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "height": ("INT", {"default": 720}),
                "width": ("INT", {"default": 1280}),
                "num_frames": ("INT", {"default": 45}),
                "num_inference_steps": ("INT", {"default": 6}),
                "guidance_scale": ("FLOAT", {"default": 1.0}),
                "flow_shift": ("INT", {"default": 17}),
                "seed": ("INT", {"default": 1024}),
                "fps": ("INT", {"default": 24}),
            }
        }

    RETURN_TYPES = ("INFERENCE_ARGS",)
    RETURN_NAMES = ("inference_args",)
    FUNCTION = "set_args"
    CATEGORY = "fastvideo"

    def set_args(
        self,
        height,
        width,
        num_frames,
        num_inference_steps,
        guidance_scale,
        flow_shift,
        seed,
        fps,
    ):
        def auto_to_none(value):
            return None if value == -99999 else value

        args = {
            "height": auto_to_none(height),
            "width": auto_to_none(width),
            "num_frames": auto_to_none(num_frames),
            "num_inference_steps": auto_to_none(num_inference_steps),
            "guidance_scale": auto_to_none(guidance_scale),
            "flow_shift": auto_to_none(flow_shift),
            "seed": auto_to_none(seed),
            "fps": auto_to_none(fps),
        }

        return(args,)

