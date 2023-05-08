# huggingface login
#from huggingface_hub import login

#login()


# load textencorder in 8bit quantized
from transformers import T5EncoderModel
from diffusers import DiffusionPipeline
import datetime
import gc
import torch
from diffusers.utils import pt_to_pil
import tkinter
import random

def flush():
    gc.collect()
    torch.cuda.empty_cache()

default_prompt = "anime girl, silver hair and blue eyes, smile"
default_negative_prompt = "lowres, bad anatomy, bad hands, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, blurry"
default_steps = 1

def winterpear(prompt, negative_prompt, steps, fullauto):
    lv2run = True
    lv3run = False
    if int(fullauto) == 1 :
        lv3run = True
    #prompt = input("prompt: ")
    if(prompt == ""):prompt = default_prompt
    elif(prompt == "exit()"):exit()
    #negative_prompt = input("negative_prompt: ")
    if(negative_prompt == ""):negative_prompt = default_negative_prompt
    elif(negative_prompt == " "):negative_prompt = ""
    #steps = input(f"stage1 steps(default: {default_steps}): ")
    if(steps == "" or int(steps) > 2147483647 or int(steps) < -1):steps = default_steps
    elif(int(steps) == -1):steps = random.randrange(0,2147483647)
    else:steps = int(steps)

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
    generator = torch.Generator().manual_seed(steps)
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds, 
        output_type="pt",
        generator=generator,
        num_inference_steps=27,
    ).images
    print("finish!")

    #apply watermarks

    pil_image = pt_to_pil(image)
    pipe.watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

    pil_image[0].save(f"./img/{start_dt}_DeepFloyd_IFstage1.png")
    # unload stage1 model
    del pipe
    flush()

    if lv2run :
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
            num_inference_steps=27,
        ).images
        print("finish!")
        pil_image = pt_to_pil(image)
        pipe.watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

        pil_image[0].save(f"./img/{start_dt}_DeepFloyd_IFstage2.png")

        #unload stage2 model
        del pipe
        flush()

    if lv2run and lv3run :
    # load stage3 model
        print("Loading stage3 model...",end="")
        pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", 
        torch_dtype=torch.float16, 
        device_map="auto"
        )
        print("finish!")

        pil_image = pipe(prompt, generator=generator, image=image, num_inference_steps=40).images

        #apply watermarks
        from diffusers.pipelines.deepfloyd_if import IFWatermarker

        watermarker = IFWatermarker.from_pretrained("DeepFloyd/IF-I-XL-v1.0", subfolder="watermarker")
        watermarker.apply_watermark(pil_image, pipe.unet.config.sample_size)

        pil_image[0].save(f"./img/{start_dt}_DeepFloyd_IFstage3.png")
        #del watermarker
        flush()

def guirun():
    prompt = e1.get("1.0",'end-1c')
    negative_prompt = e2.get("1.0",'end-1c')
    st = e3.get()
    fullauto = servervar.get()
    winterpear(prompt, negative_prompt, st, fullauto)


window=tkinter.Tk()
window.title("WinterPear-3000")
#window.geometry("600x200")
window.resizable(True, True)
servervar = tkinter.StringVar(None, False)


if __name__ == '__main__': 
    label_prompt = tkinter.Label(window, text="prompt")
    label_negative = tkinter.Label(window, text="negative\nprompt")
    label_st = tkinter.Label(window, text="seed (-1 = ran)")
    labelserver = tkinter.Label(window, text="stage 2 or 3")
    button = tkinter.Button(window, text="run", overrelief="solid", width=10, command=guirun)
    e1 = tkinter.Text(window, height=10)
    e2 = tkinter.Text(window, height=10)
    e3 = tkinter.Entry(window)
    label_prompt.grid(row=0)
    e1.grid(row=0, column=1, columnspan=5)
    label_negative.grid(row=1)
    e2.grid(row=1, column=1, columnspan=5)
    label_st.grid(row=2)
    e3.grid(row=2, column=1, columnspan=5)
    labelserver.grid(row=3)
    button.grid(row=4, columnspan=5)
    R0 = tkinter.Radiobutton(window, text="2", variable=servervar, value=0)
    R0.grid(row=3, column=1)
    R1 = tkinter.Radiobutton(window, text="3", variable=servervar, value=1)
    R1.grid(row=3, column=2)
    window.mainloop()
