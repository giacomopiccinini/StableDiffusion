from __future__ import annotations

import io
import os
from pathlib import Path

import modal
import typer

from unique_names_generator import get_random_name

# All Modal programs need a stub
stub = modal.Stub("stable-diffusion-cli")

# We will be using `typer` to create CLI interface
app = typer.Typer()

# Options we will use throughout the script
MODEL_ID = "runwayml/stable-diffusion-v1-5"
CACHE_PATH = "/vol/cache"


def download_models():
    
    """Download necessary models and schedulers from HuggingFace"""

    import diffusers
    import torch

    # Fetch HF token
    hugging_face_token = os.environ["HUGGINGFACE_TOKEN"]

    # Download scheduler configuration. Experiment with different schedulers
    # to identify one that works best for your use-case
    scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(
        MODEL_ID,
        subfolder="scheduler",
        use_auth_token=hugging_face_token,
        cache_dir=CACHE_PATH,
    )
    scheduler.save_pretrained(CACHE_PATH, safe_serialization=True)

    # Downloads Stable Diffusion pipeline
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        use_auth_token=hugging_face_token,
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=CACHE_PATH,
    )
    pipe.save_pretrained(CACHE_PATH, safe_serialization=True)


# Create image (Docker-like) to be used by Modal backend
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate",
        "diffusers[torch]>=0.10",
        "ftfy",
        "torch",
        "torchvision",
        "transformers",
        "triton",
        "safetensors",
        "unique_names_generator"
    )
    .pip_install("xformers", pre=True)
    .run_function(
        download_models,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

# Assign this image to the stub we create before
stub.image = image


# Create a class for Stable Diffusion
class StableDiffusion:
    def __enter__(self):
        
        """Magic __enter__ method to create the environment, minimising time loss due to
        image recreation"""

        import diffusers
        import torch

        # Allow TF32 to get a significant speed-up
        torch.backends.cuda.matmul.allow_tf32 = True

        # Create Scheduler
        scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(
            CACHE_PATH,
            subfolder="scheduler",
            solver_order=2,
            prediction_type="epsilon",
            thresholding=False,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            denoise_final=True
        )

        # Define pipeline
        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            CACHE_PATH, scheduler=scheduler
        ).to("cuda")

        # Memory efficiency
        self.pipe.enable_xformers_memory_efficient_attention()

    # Define it as a function that will be used by Modal backend.
    # The particular type of GPU could be different if needed
    @stub.function(gpu="A10G")
    def run_inference(
        self, prompt: str, steps: int = 20, batch_size: int = 4
    ) -> list[bytes]:
        """Create images using Stable Diffusion"""
        import torch

        with torch.inference_mode():
            with torch.autocast("cuda"):
                # Run SD and extract images
                images = self.pipe(
                    [prompt] * batch_size,
                    num_inference_steps=steps,
                    guidance_scale=7.0,
                ).images

        # Convert to PNG bytes
        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())
        return image_output


@stub.local_entrypoint
def entrypoint(prompt: str, samples: int = 2, steps: int = 50, batch_size: int = 1):
    
    """ Generate and save images """
    
    # Random name identifier 
    unique_name = get_random_name(separator="-")

    # CLI printing our choices
    typer.echo(
        f"\n \
            Identifier: {unique_name} \n \
                Prompt: {prompt} \n \
                    Steps: {steps} \n \
                        Samples: {samples} \n \
                            Batch: {batch_size}"
    )

    # Define (and create if not existing) directory where images are stored
    output_directory = Path(f"Output/{unique_name}")
    if not output_directory.exists():
        output_directory.mkdir(exist_ok=True, parents=True)
        
    # Save prompt to output directory
    with open(f"{output_directory}/prompt.txt", "w") as f:
        f.write(prompt)

    # Instantiate Stable Diffusion object
    sd = StableDiffusion()
    
    # SD generate "samples" images
    for sample in range(samples):
        
        # Run inference
        images = sd.run_inference.call(prompt, steps, batch_size)

        for batch, image_bytes in enumerate(images):
            
            # Create output path
            output_path = f"{output_directory}/image_{batch}_{sample}.png"
            
            # Save images to file
            with open(output_path, "wb") as f:
                f.write(image_bytes)
