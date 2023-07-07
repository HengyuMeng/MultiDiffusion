from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    """
    固定窗口大小为64×64，滑动窗口每次滑动8个单位，计算整个全景图的方框个数以及位置
    """
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views


class MultiDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad() # 将函数内的操作设置为不进行梯度计算
    def get_text_embeds(self, prompt, negative_prompt):
        """
        接受正向和负向文本作为输入，并使用 tokenizer 对文本进行分词，
        然后将分词后的输入张量传递给 text_encoder 模型进行嵌入处理。
        最后，将正向和负向文本的嵌入结果连接在一起，并返回最终的文本嵌入表示
        """
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        
        # 使用 tokenizer 对正向文本进行分词
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        
        # 将分词后的输入张量传递给 text_encoder 模型，进行嵌入处理，并取得嵌入结果的第一个（0 索引）张量
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        # 将正向和负向文本的嵌入结果连接起来，形成最终的文本嵌入表示
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        # 通过给定的潜在向量，模型将生成一个对应的图像样本
        imgs = self.vae.decode(latents).sample
        
        # 将图像像素值除以 2 并加上 0.5，将像素值的范围从 [-1, 1] 映射到 [0, 1]。
        # 然后，使用 .clamp(0, 1) 方法将图像像素值限制在 [0, 1] 的范围内，确保图像的像素值不超出该范围
        imgs = (imgs / 2 + 0.5).clamp(0, 1) 
        return imgs

    @torch.no_grad()
    def text2panorama(self, prompts, negative_prompts='', height=512, width=2048, num_inference_steps=50,
                      guidance_scale=7.5):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # Define panorama grid and get views
        # 这一步，我们将我们自定义的噪声传入作为初始的噪声
        latent = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)
        views = get_views(height, width)

        # 创建了与全景图像张量 latent 相同形状的计数和值张量。
        # 计数张量 count 用于累计生成过程中每个像素点的更新次数，
        # 值张量 value 用于存储生成过程中每个像素点的累计值
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                count.zero_()
                value.zero_()

                for h_start, h_end, w_start, w_end in views:
                    # TODO we can support batches, and pass multiple views at once to the unet
                    # 提取当前窗口的的潜在向量
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end]

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latent_view] * 2)

                    # predict the noise residual
                    # 使用 self.unet 模型对扩展的潜在向量 latent_model_input 进行前向传播，预测噪声残差。
                    # 其中，t 是当前推理步数，encoder_hidden_states 是文本嵌入表示
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

                    # perform guidance
                    # 先将噪声残差张量 noise_pred 分成两个部分，分别赋值给 noise_pred_uncond 和 noise_pred_cond。
                    # 然后，通过将无条件部分 noise_pred_uncond 和有条件部分 noise_pred_cond 进行加权叠加，计算得到最终的噪声残差张量
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    # 生成过程中使用条件和无条件的噪声，以提供更多的控制和指导。具体的权重由 guidance_scale 参数控制
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # compute the denoising step with the reference model
                    latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']

                    # 将去噪后的潜在向量 latents_view_denoised 和计数值 1 累加到值张量 value 和计数张量 count 的对应视图位置上。
                    # 在每个像素点上都会累计多次去噪的结果和计数
                    value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                    count[:, :, h_start:h_end, w_start:w_end] += 1

                # take the MultiDiffusion step
                # 对于整个latent，对于计数不为零的像素点，将值张量 value 除以计数张量 count，得到平均值；
                # 对于计数为零的像素点，保持值张量 value 不变。最终，更新全景图像的潜在向量 latent
                latent = torch.where(count > 0, value / count, value)

        # Img latents -> imgs
        # 将更新后的全景图像潜在向量 latent 作为输入，进行解码，得到图像张量 
        imgs = self.decode_latents(latent)  # [1, 3, 512, 512]
        # 对生成的图像进行后处理，将图像转换为 PIL 图像格式，并将其作为函数的返回值
        img = T.ToPILImage()(imgs[0].cpu())
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='a photo of the dolomites')
    parser.add_argument('--negative', type=str, default='')
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'],
                        help="stable diffusion version")
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = MultiDiffusion(device, opt.sd_version)

    img = sd.text2panorama(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # save image
    img.save('out.png')
