from diffusers.pipeline_utils import DiffusionPipeline
import torch
import torch.nn.functional as F
from typing import Optional, Union, List
from torch.nn import MSELoss
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
            record_process = False
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
            model_output = self.unet(images, t).sample
            
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
    )-> tuple:
        recovered = self.generate_from_scratch(
                input_images=input_images,
                generator=generator,
                time_steps=time_steps
        )[0]
        
        mse = F.mse_loss(recovered, input_images, reduction = None)
        
        return mse
    
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
                model_output = self.unet(images, t).sample
                mse = F.mse_loss(noise, model_output, reduction=None)
                mse.backward()
                grads += images_active.grad

        #calculate heatmap
        pixel_dim = list(range(2, len(input_images.shape)))
        alpha = torch.mean(grads, pixel_dim, keepdims=True)
        grad_shape = input_images.shape[2:]
        alpha = alpha.repeat(1,1,*grad_shape)
        heat_map = F.relu((alpha * input_images).sum(dim = 1, keepdims=True))
        
        return heat_map        
        