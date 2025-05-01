class DITConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "precision": (["fp16", "bf16"], {"default": "fp16"}),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, precision, **kwargs):
        # Handle None value for scale_factor
        if precision == -99999 or precision == None:
            # This is valid - we'll use the default value in the set_args method
            return True

    RETURN_TYPES = ("DIT_CONFIG",)
    RETURN_NAMES = ("dit_config",)
    FUNCTION = "set_args"
    CATEGORY = "fastvideo"

    def set_args(
        self,
        precision
    ):
        def auto_to_none(value):
            return None if value == -99999 else value

        args = {
            "precision": auto_to_none(precision),
        }

        return(args,)
