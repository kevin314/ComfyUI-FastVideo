class VAEConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "scale_factor": ("INT", {"default": 8}),
                "vae_sp": ([True, False], {"default": True}),
                "tiling": ([True, False], {"default": True}),
                "precision": (["fp16", "bf16"], {"default": "fp16"}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, scale_factor=None, **kwargs):
        # Handle None value for scale_factor
        if scale_factor is None:
            # This is valid - we'll use the default value in the set_args method
            return True
        
        # For non-None values, ensure it's a valid integer
        try:
            int(scale_factor)
            return True
        except (ValueError, TypeError):
            return f"scale_factor must be an integer, got {type(scale_factor).__name__}"

    RETURN_TYPES = ("VAE_CONFIG",)
    RETURN_NAMES = ("vae_config",)
    FUNCTION = "set_args"
    CATEGORY = "fastvideo"

    def set_args(
        self,
        scale_factor,
        vae_sp,
        tiling,
        precision
    ):
        def auto_to_none(value):
            return None if value == -99999 else value
        
        args = {
            "scale_factor": auto_to_none(scale_factor),
            "vae_sp": auto_to_none(vae_sp),
            "tiling": auto_to_none(tiling),
            "precision": auto_to_none(precision),
        }

        return(args,)
