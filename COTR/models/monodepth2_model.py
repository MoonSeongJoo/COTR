import os
import torch
from monodepth2.networks.resnet_encoder import ResnetEncoder
from monodepth2.networks.depth_decoder import DepthDecoder  

class MonoDepth():
    
    def __init__(self):
        self.model_name         = "mono_resnet50_640x192"
        self.encoder_path       = os.path.join("/home/seongjoo/work/autocalib/COTR/monodepth2/models", self.model_name, "encoder.pth")
        self.depth_decoder_path = os.path.join("/home/seongjoo/work/autocalib/COTR/monodepth2/models", self.model_name, "depth.pth")
        
        # device = torch.device("cuda")
        self.encoder = ResnetEncoder(50, False)
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        
        # self.loaded_dict_enc = torch.load(self.encoder_path, map_location=device)
        self.loaded_dict_enc = torch.load(self.encoder_path, map_location='cuda')
        self.filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(self.filtered_dict_enc)
        self.encoder.cuda()
        # self.encoder.to(device)
        # print ('encoder device : ' , next(self.encoder.parameters()).device)

        # self.loaded_dict = torch.load(self.depth_decoder_path, map_location=device)
        self.loaded_dict = torch.load(self.depth_decoder_path, map_location='cuda')
        self.depth_decoder.load_state_dict(self.loaded_dict)
        # self.depth_decoder.to(device)
        self.depth_decoder.cuda()
        # print ('decoder device : ' , next(self.depth_decoder.parameters()).device)
        
        self.encoder.eval()
        self.depth_decoder.eval()
    
    def forward(self, rgb_input):
        
        with torch.no_grad():
            rgb_features = self.encoder(rgb_input)
            rgb_outputs  = self.depth_decoder(rgb_features)
            
        rgb_depth_pred = rgb_outputs[("disp", 0)]
        
        return rgb_depth_pred

# class monodepth_encoder():
    
#     def __init__(self, network_name):
#         self.encoder_path = os.path.join("/root/work/COTR/monodepth2/models", network_name, "encoder.pth")
#         self.encoder = ResnetEncoder(50, False)
#         self.loaded_dict_enc = torch.load(self.encoder_path, map_location='cuda')
#         self.filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
#         self.encoder.load_state_dict(self.filtered_dict_enc)
#         self.encoder.cuda()      

# class monodepth_decoder():
#     def __init__(self, network_name):
#         self.depth_decoder_path = os.path.join("/root/work/COTR/monodepth2/models", network_name, "depth.pth")
#         self.depth_decoder = DepthDecoder(num_ch_enc=[64, 256, 512, 1024, 2048], scales=range(4))
#         self.loaded_dict = torch.load(self.depth_decoder_path, map_location='cuda')
#         self.depth_decoder.load_state_dict(self.loaded_dict)
#         self.depth_decoder.cuda()

# def build_monodepth_model():
#     model_name = "mono_resnet50_640x192"
#     encoder = monodepth_encoder(model_name)
#     decoder = monodepth_decoder(model_name)
#     model = MonoDepth(encoder,decoder)
#     return model