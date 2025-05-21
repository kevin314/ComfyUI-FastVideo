# ComfyUI-FastVideo

A custom node for ComfyUI that provides fast video generation capabilities using the FastVideo model. This node makes it easy to integrate high-quality text-to-video generation into your ComfyUI workflows.

![FastVideo Example](https://raw.githubusercontent.com/PLACEHOLDER/ComfyUI-FastVideo/main/examples/example.gif)

## Features

- Generate high-quality videos from text prompts
- Configurable video parameters (resolution, frame count, FPS)
- Support for multiple GPUs with tensor and sequence parallelism
- Advanced configuration options for VAE, Text Encoder, and DIT components
- Interruption/cancellation support for long-running generations
- Seamless integration with ComfyUI workflows

## Installation

### Prerequisites

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed and working
- FastVideo model installed (see below)
- CUDA-capable GPU(s) with sufficient VRAM

### Install using ComfyUI Manager

1. Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) if you haven't already
2. Find and install "ComfyUI-FastVideo" through the Manager interface

### Manual Installation

1. Clone this repository into your ComfyUI custom_nodes directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/PLACEHOLDER/ComfyUI-FastVideo
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Model Installation

Download the FastVideo model from [HuggingFace](https://huggingface.co/PLACEHOLDER/FastHunyuan-diffusers) and place it in an appropriate location (default path is set to "FastVideo/FastHunyuan-diffusers" in your workspace).

## Usage

After installation, the following nodes will be available in the ComfyUI interface under the "fastvideo" category:

- **Video Generator**: The main node for generating videos from prompts
- **Inference Args**: Configure video generation parameters
- **VAE Config**: Fine-tune VAE settings
- **Text Encoder Config**: Configure text encoder settings
- **DIT Config**: Adjust DIT model parameters
- **Load Image Path**: Load images for potential conditioning

### Basic Workflow

1. Add a "Video Generator" node to your workflow
2. Connect optional configuration nodes (Inference Args, VAE Config, etc.)
3. Set your prompt and output path
4. Configure GPU settings based on your hardware
5. Execute the workflow to generate a video

### Node Configuration

#### Video Generator

- **prompt**: Text description of the video to generate
- **output_path**: Directory where generated videos will be saved
- **num_gpus**: Number of GPUs to use for generation
- **model_path**: Path to the FastVideo model
- **embedded_cfg_scale**: Classifier-free guidance scale
- **sp_size**: Sequence parallelism size
- **tp_size**: Tensor parallelism size
- **precision**: Model precision (fp16 or bf16)

#### Inference Args

- **height/width**: Resolution of the output video
- **num_frames**: Number of frames to generate
- **num_inference_steps**: Number of diffusion steps per frame
- **guidance_scale**: Classifier-free guidance scale
- **flow_shift**: Frame flow shift parameter
- **seed**: Random seed for reproducible generation
- **fps**: Frames per second of the output video
- **image_path**: Optional path to input image for conditioning

## Examples

### Simple Text to Video

[Screenshot of simple workflow setup]

### Advanced Configuration

[Screenshot of complex workflow]

## Performance Tips

- Use multiple GPUs when available for faster generation
- Adjust tensor and sequence parallelism based on your GPU count and VRAM
- Lower resolution and frame count for quicker previews
- Use fp16 precision for optimal speed/quality balance

## Known Issues and Limitations

- High VRAM requirements for high-resolution generation
- Generation can be slow on consumer-grade hardware
- [Any other known issues]

## License

This project is licensed under [LICENSE file included in repository]

## Acknowledgements

- FastVideo model creators
- ComfyUI team for the excellent framework
- [Other acknowledgements as needed]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request