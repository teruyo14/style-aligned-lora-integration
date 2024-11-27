# Style-Aligned Image Generation with LoRA and Gradio

This repository contains a **Google Colab** notebook that demonstrates how to generate images using the **Style-Aligned** method integrated with **LoRA** models and a **Gradio** interface. The notebook allows for interactive image generation using custom prompts and provides fine-grained control over the style and content of the generated images.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Open the Colab Notebook](#open-the-colab-notebook)
- [Usage](#usage)
  - [Replacing Dummy Paths](#replacing-dummy-paths)
  - [Running the Notebook](#running-the-notebook)
  - [Using the Gradio Interface](#using-the-gradio-interface)
- [Notes](#notes)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

The notebook demonstrates how to:

- Integrate the **Style-Aligned** method with the **Stable Diffusion XL** pipeline.
- Load **LoRA** models from `.safetensors` files.
- Use **Gradio** to create an interactive web interface for image generation.
- Customize prompts, image dimensions, and model parameters for fine-grained control.

**Important:** The notebook contains **dummy paths** that need to be replaced with actual model paths. Please ensure you have the necessary models and weights before running the notebook.

## Features

- **Style-Aligned Method**: Aligns the style of generated images for better control and consistency.
- **LoRA Integration**: Utilizes Low-Rank Adaptation (LoRA) for efficient fine-tuning.
- **Gradio Interface**: Provides an interactive UI for real-time image generation.
- **Customizable Parameters**: Adjust prompts, alpha values, seeds, and image dimensions.

## Getting Started

### Prerequisites

- **Google Colab** account ([colab.research.google.com](https://colab.research.google.com/))
- Basic understanding of Python and machine learning models
- Access to the required models and weights:
  - Base model (e.g., Stable Diffusion XL)
  - Textual inversion embeddings
  - LoRA models in `.safetensors` format



### Open the Colab Notebook

- Navigate to the cloned repository.
- Open the `.ipynb` notebook file in Google Colab.

Alternatively, you can open the notebook directly in Colab using the following link:

```
https://drive.google.com/file/d/1bINJ8oA3zTSWLr4DF5-_jdaAHysuuiba/view?usp=sharing
```

## Usage

### Replacing Dummy Paths

The notebook contains **dummy paths** that need to be replaced with actual paths to your models and weights.

#### Example Dummy Paths

- `"path/to/your/base-model"`
- `"path/to/your/textual-inversion.safetensors"`
- `"path/to/your/lora-model.safetensors"`

#### Steps to Replace Paths

1. Locate the `initialize_pipeline` function in the notebook.
2. Replace the dummy paths with the actual paths where your models and weights are stored.
   - **Base Model**: Provide the path or name of your base model (e.g., `"stabilityai/stable-diffusion-xl-base-1.0"`).
   - **Textual Inversion Embeddings**: Provide the path to your `.safetensors` file containing the embeddings.
   - **LoRA Model**: Provide the path to your LoRA model in `.safetensors` format.

**Note:** Ensure that the models are accessible and compatible with the notebook's code.

### Running the Notebook

1. **Install Dependencies**

   The notebook includes commands to clone the Style-Aligned repository and install required packages.

   ```python
   !git clone https://github.com/google/style-aligned
   !mv style-aligned/* .
   !pip install -r requirements.txt
   !pip install gradio controlnet_aux -q
   ```

2. **Import Libraries**

   The necessary libraries are imported at the beginning of the notebook.

   ```python
   import torch
   from diffusers import StableDiffusionXLPipeline, DDIMScheduler
   from safetensors.torch import load_file
   import sa_handler
   import gradio as gr
   ```

3. **Run Cells Sequentially**

   - Execute each cell one by one to set up the environment.
   - Ensure that the paths are correctly set before running the `initialize_pipeline` function.

### Using the Gradio Interface

After setting up the pipeline, the notebook launches a Gradio interface.

```python
demo.launch(share=True)
```

#### Interface Overview

- **Prompt**: Input your custom text prompt for image generation.
- **LoRA Alpha Value**: Adjust the influence of the LoRA model (range from 0.0 to 1.0).
- **Seed**: Set a seed for reproducibility (optional).
- **Width & Height**: Specify the dimensions of the generated images.
- **Generate Images**: Click to generate images based on the provided parameters.
- **Generated Images**: View the output images in the Gradio gallery.

#### Steps to Generate Images

1. **Enter Prompt**

   Provide a descriptive prompt, e.g., `"1girl, solo, look at viewer, face close up, best quality"`.

2. **Adjust Parameters**

   - **LoRA Alpha Value**: Set to `0.3` for a balanced effect.
   - **Seed**: Use `0` for a random seed or specify a number.
   - **Width & Height**: Default is `512x960`; adjust as needed.

3. **Generate Images**

   Click the **Generate Images** button to start the image generation process.

4. **View Results**

   The generated images will appear in the gallery below the interface.

## Notes

- **Model Compatibility**: Ensure that all models and weights are compatible with each other and the code.
- **Performance**: The notebook is designed to run on GPU. Running on CPU may be significantly slower.
- **Resource Usage**: High-resolution image generation can be resource-intensive. Adjust image dimensions if you encounter memory issues.
- **Gradio Sharing**: The `share=True` parameter in `demo.launch()` allows you to share the Gradio interface via a public link.

## Citation

If you use this code or find it helpful, please cite the original paper:

```bibtex
@article{style-aligned-2023,
  title={Style Aligned Techniques for High-Resolution Image Generation},
  author={Authors},
  journal={arXiv preprint arXiv:2312.02133},
  year={2023}
}
```

## Acknowledgements

- [Google's Style-Aligned Repository](https://github.com/google/style-aligned/)
- [Stable Diffusion XL](https://github.com/Stability-AI/stablediffusion)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Gradio](https://gradio.app/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
