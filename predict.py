import os
import torch
from cog import BasePredictor, Input, Path
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        ckpt_dir = './weights/Kolors'

        self.text_encoder = ChatGLMModel.from_pretrained(
            f'{ckpt_dir}/text_encoder',
            torch_dtype=torch.float16).half()
        self.tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
        self.vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
        self.scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
        self.unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()

        self.pipe = StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            force_zeros_for_empty_prompt=False)
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

    def predict(
        self,
        prompt: str = Input(description="Text prompt for image generation"),
        height: int = Input(description="Height of the output image", default=1024),
        width: int = Input(description="Width of the output image", default=1024),
        num_images: int = Input(description="Number of images to generate", default=1),
        steps: int = Input(description="Number of inference steps", default=50),
        seed: int = Input(description="Random seed for generation (optional)", default=None)
    ) -> Path:
        """Run a single prediction on the model"""
        if height <= 0 or width <= 0:
            raise ValueError("Height and width must be positive integers")
        if num_images <= 0:
            raise ValueError("Number of images must be a positive integer")
        if steps <= 0:
            raise ValueError("Number of steps must be a positive integer")

        generator = torch.Generator(self.pipe.device)
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator.manual_seed(seed)

        images = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=5.0,
            num_images_per_prompt=num_images,
            generator=generator).images

        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []

        for i, image in enumerate(images):
            output_file = f"{output_dir}/output_{i+1}.png"
            image.save(output_file)
            output_paths.append(output_file)

        print(f"Seed used: {seed}")

        # Return the path to the first image
        return Path(output_paths[0])
