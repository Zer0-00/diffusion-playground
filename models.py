from diffusers.pipeline_utils import DiffusionPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple


from dataclasses import dataclass


from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block

class AnomalyDetectionModel(DiffusionPipeline):
    def __init__(self, unet, scheduler):

        self.register_modules(unet=unet, scheduler=scheduler)
    
    @torch.no_grad()
    def __call__(
            self,
            input_images: torch.Tensor,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            time_steps: Optional[Union[int, torch.Tensor]] = 1000,
            record_process = False
    )-> tuple:
        pass
        
    @torch.no_grad()
    def generate_from_scratch(
            self,
            input_images: torch.Tensor,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            time_steps: Optional[Union[int, torch.Tensor]] = 1000,
            record_process = False,
            model_kwargs = dict()
    )-> tuple:
        """
        Parameters:
            input_image: input image(N,C,H,W)
            generator: generator for controlling inference seed
            num_noise_steps: number of noising steps(<= scheduler.config.num_train_steps)
        """ 
        if not torch.is_tensor(time_steps):
            time_steps = torch.tensor(time_steps).to(self.device)
        
        input_images = input_images.to(self.device)
        noise = torch.randn(input_images.shape).to(input_images.device)
        images = self.scheduler.add_noise(input_images, noise, time_steps)
        
        if record_process:
            record = []
        
        for t in range(time_steps,-1, -1):
            # predict noise
            model_output = self.unet(images, t, **model_kwargs).sample
            
            # compute denoising
            images = self.scheduler.step(model_output, t, images, generator=generator).prev_sample
            
            #save intermediate results
            if record_process:
                record.append(images)
            
        return (images,) if not record_process else (images,record)
    
    @torch.no_grad()
    def generate_mse_detection_map(
            self,
            input_images: torch.Tensor,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            time_steps: Optional[Union[int, torch.Tensor]] = 1000,
            return_generated = False
    )-> tuple:
        recovered = self.generate_from_scratch(
                input_images=input_images,
                generator=generator,
                time_steps=time_steps
        )[0]
        
        mse = F.mse_loss(recovered, input_images, reduction = "none")
        heat_map = mse.sum(dim=1, keepdim=True)
        
        if return_generated:
            return heat_map, recovered
        return heat_map
    
    def generate_grad_detection_map(
            self,
            input_images: torch.Tensor,
            time_steps: List[int] = 1000,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            consecutive = False
    ):
        """

        Args:
            input_images (torch.Tensor): input images of (B, C, H, W)
            time_steps (List[int], optional): timestep of noising. Defaults to 1000.
            generator (Optional[Union[torch.Generator, List[torch.Generator]]], optional): generator used for controlling output. Defaults to None.
            consecutive (bool, optional): whether using consecutive t. Defaults to False.

        Returns:
            heatmap
        """
        self.unet.eval()
        
        time_steps = torch.tensor(time_steps).to(self.device)
        
        #now we are trying seperately generate images instead of using a consecutive generation process
        #TODO: try to use a consecutive generation process from [400,600]
        grads = torch.zeros(input_images.shape, device=self.device)
        if not consecutive:
            for t in time_steps:
                images_active = input_images.to(self.device)
                images_active.requires_grad = True
                noise = torch.randn(images_active.shape, device=self.device)
                images = self.scheduler.add_noise(images_active, noise, t)
                model_output = self.unet(images, t)
                mse = F.mse_loss(noise, model_output, reduction="none")
                mse.backward()
                grads += images_active.grad
        else:
            pass    

        #calculate heatmap
        pixel_dim = list(range(2, len(input_images.shape)))
        alpha = torch.mean(grads, pixel_dim, keepdims=True)
        grad_shape = input_images.shape[2:]
        alpha = alpha.repeat(1,1,*grad_shape)
        heat_map = F.relu((alpha * input_images).sum(dim = 1, keepdims=True))
        
        return heat_map        


@dataclass
class ClassGuidedUNet2DOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    sample: torch.FloatTensor


class ClassGuidedUNet2DModel(ModelMixin, ConfigMixin):
    r"""
    Adapt from diffusers.Unet2DModel
    
    ClassGuidedUNet2DModel is a 2D UNet model that takes in a noisy sample ,a timestep and a class label and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        time_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of time embedding to use.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to :
            obj:`True`): Whether to flip sin to cos for fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`): Tuple of downsample block
            types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            The mid block type. Choose from `UNetMidBlock2D` or `UnCLIPUNetMidBlock2D`.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(224, 448, 672, 896)`): Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for the normalization.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for the normalization.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for resnet blocks, see [`~models.resnet.ResnetBlock2D`]. Choose from `default` or `scale_shift`.
        num_class('int', default to 2):
            number of class
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        act_fn: str = "silu",
        attention_head_dim: int = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        num_class = 2
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time and class
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        
        self.class_embedding = nn.Embedding(num_class, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
            add_attention=add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=attention_head_dim,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels:torch.Tensor,
        return_dict: bool = True,
    ) -> Union[ClassGuidedUNet2DOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): (batch) timesteps
            class_labels("torch.FloatTensor"): (batch) class labels
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.ClassGuidedUNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~model.ClassGuidedUNet2DOutput`] or `tuple`: [`~models.unet_2d.UNet2DOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time and class
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        
        class_emb = self.class_embedding(class_labels)
        emb = self.time_embedding(t_emb) + class_emb

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return ClassGuidedUNet2DOutput(sample=sample)
