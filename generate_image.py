from diffusers.pipeline_utils import DiffusionPipeline
import torch

class anomaly_detection_model(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().init()
        self.register_modules(unet=unet, schuduler=scheduler)
        
    @torch.no_grad()
    def __call__(
        self,
    ):
        pass