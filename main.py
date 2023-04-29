# huggingface login
from huggingface_hub import login

login()


# load textencorder in 8bit quantized
from transformers import T5EncoderModel
from diffusers import DiffusionPipeline
import datetime
import gc
import torch
from diffusers.utils import pt_to_pil

def flush():
    gc.collect()
    torch.cuda.empty_cache()

default_prompt = "anime girl, silver hair and blue eyes, smile"
default_negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
default_steps = 50

while(True):
    prompt = input("prompt: ")
    if(prompt == ""):prompt = default_prompt
    elif(prompt == "exit()"):break
    negative_prompt = input("negative_prompt: ")
    if(negative_prompt == ""):negative_prompt = default_negative_prompt
    steps = input(f"stage1 steps(default: {default_steps}): ")
    if(steps == ""):steps = default_steps
    start_dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


    print("Loading Text Encoder...",end="")
    text_encoder = T5EncoderModel.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0",
        subfolder="text_encoder", 
        device_map="auto", 
        load_in_8bit=True, 
        variant="8bit"
    )
    print("finish!")

    # load stage1 model for text encording
    print("Encording Text...",end="")
    pipe = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0", 
        text_encoder=text_encoder, # pass the previously instantiated 8bit text encoder
        unet=None, 
        device_map="auto"
    )
    prompt_embeds, negative_embeds = pipe.encode_prompt(prompt=prompt, negative_prompt=negative_prompt)
    print("finish!")

    # unload textencorder
    del text_encoder
    del pipe

    flush()
    # load stage1 model for generation
    print("Loading stage1 model...",end="")
    pipe = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0", 
        text_encoder=None, 
        variant="fp16", 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    print("finish!")
    print("Generating images...",end="")
    generator = torch.Generator().manual_seed(1)
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds, 
        output_type="pt",
        generator=generator,
        num_inference_steps=steps,
    ).images
    print("finish!")

    #apply watermarks

    pil_image = pt_to_pil(image)
    pipe.watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

    pil_image[0].save(f"./{start_dt}_DeepFloyd_IFstage1.png")
    """
    restart_check = input("Restart?: ")
    if not(restart_check == "no" or restart_check == "NO" or restart_check == "N"):
        continue
    """
    # unload stage1 model
    del pipe
    flush()

    # load stage2 model
    print("Loading stage2 model...",end="")
    pipe = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0", 
        text_encoder=None, # no use of text encoder => memory savings!
        variant="fp16", 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    print("Generating...")
    image = pipe(
        image=image, 
        prompt_embeds=prompt_embeds, 
        negative_prompt_embeds=negative_embeds, 
        output_type="pt",
        generator=generator,
    ).images
    print("finish!")
    pil_image = pt_to_pil(image)
    pipe.watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

    pil_image[0].save(f"./{start_dt}_DeepFloyd_IFstage2.png")
    """
    restart_check = input("Restart?: ")
    if not(restart_check == "no" or restart_check == "NO" or restart_check == "N"):
        continue
    """

    #unload stage2 model
    del pipe
    flush()

    # load stage3 model
    print("Loading stage3 model...",end="")
    pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", 
    torch_dtype=torch.float16, 
    device_map="auto"
    )
    print("finish!")

    pil_image = pipe(prompt, generator=generator, image=image).images

    #apply watermarks
    from diffusers.pipelines.deepfloyd_if import IFWatermarker

    watermarker = IFWatermarker.from_pretrained("DeepFloyd/IF-I-XL-v1.0", subfolder="watermarker")
    watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

    pil_image[0].save(f"./{start_dt}_DeepFloyd_IFstage3.png")
    del watermarker
    flush()
