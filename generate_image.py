from diffusers.pipeline_utils import DiffusionPipeline
import torch
from typing import Optional, Union, List

class AnomalyDetectionModel(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().init()
        self.register_modules(unet=unet, schuduler=scheduler)
        
    @torch.no_grad()
    def __call__(
            self,
            input_images: torch.Tensor,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            time_steps: int = 1000,
            record_process = False
    )-> tuple:
        """
        Parameters:
            input_image: input image(N,C,H,W)
            generator: generator for controlling inference seed
            num_noise_steps: number of noising steps(<= scheduler.config.num_train_steps)
        """ 
    
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
            record.append(images)
            
        return (images,) if not record_process else (images,record)
            