from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import s3prl.hub as hub

# Wav2vec2, Hubert
class UpstreamTransformer(nn.Module):
    def __init__(self, upstream_model='wav2vec2', feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model) # hubert
        
        for param in self.upstream.parameters():
            param.requires_grad = False
        
        for param in self.upstream.model.encoder.layers.parameters():
            param.requires_grad = True

        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[5:].parameters():
                param.requires_grad = True
        
        self.language_classifier = nn.Sequential(
            nn.Linear(feature_dim, 14)
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        x = torch.mean(x, dim=1)
        language = self.language_classifier(x)
        return language


# class UpstreamTransformerXLSR(nn.Module):
#     def __init__(self, upstream_model='xlsr_300m', feature_dim=1024, unfreeze_last_conv_layers=False):
#         super().__init__()

#         self.xlsrModelsUrls = {'xlsr_300m': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt',
#         'xlsr_1b': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_960m_1000k.pt',
#         'xlsr_2b': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_2B_1000k.pt'}

#         self.upstream = getattr(hub, 'wav2vec2_url')(self.xlsrModelsUrls[upstream_model])
        
#         for param in self.upstream.parameters():
#             param.requires_grad = False
        
#         for param in self.upstream.model.encoder.layers.parameters():
#             param.requires_grad = True

#         if unfreeze_last_conv_layers:
#             for param in self.upstream.model.feature_extractor.conv_layers[5:].parameters():
#                 param.requires_grad = True
        
#         self.language_classifier = nn.Sequential(
#             nn.Linear(feature_dim, 14),
#         )

#     def simple_forward(self, x, x_len):
#         x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
#         x = self.upstream(x)['last_hidden_state']
#         x = torch.mean(x, dim=1)
#         language = self.language_classifier(x)
#         return language
    
#     def forward(self, x, x_len):
#         return self.simple_forward(x, x_len)


from transformers import Wav2Vec2Processor, Wav2Vec2Model

class UpstreamTransformerXLSR(nn.Module):
    def __init__(self, upstream_model='xlsr_300m', feature_dim=1024, unfreeze_last_conv_layers=False):
        super().__init__()

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.encoder.encoder.layers.parameters():
            param.requires_grad = True

        if unfreeze_last_conv_layers:
            for param in self.encoder.model.feature_extractor.conv_layers[5:].parameters():
                param.requires_grad = True
        
        self.language_classifier = nn.Sequential(
            nn.Linear(feature_dim, 14),
        )

    def simple_forward(self, x, x_len):
        x = self.processor(x, sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0).to("cuda")
        x = self.encoder(x).last_hidden_state
        x = torch.mean(x, dim=1)
        language = self.language_classifier(x)
        return language
    
    def forward(self, x, x_len):
        return self.simple_forward(x, x_len)


# from speechbrain.pretrained import EncoderClassifier

class PretrainedLangID(nn.Module):
    def __init__(self, upstream_model=None, feature_dim=None, unfreeze_last_conv_layers=None):
        super().__init__()
        self.lang_enc = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp", run_opts={"device":"cuda"})
        
        for param in self.lang_enc.parameters():
            param.requires_grad = False
        
        print(self.lang_enc)
        # self.lang_enc.eval()
        
        self.language_classifier = nn.Sequential(
            nn.Linear(256, 14)
        )

    def forward(self, x, x_len):
        # print(x.shape)
        # x = x.squeeze()
        language_emb = self.lang_enc.encode_batch(x).squeeze(1)
        # print("SHAPE = ", language_emb.shape)
        language = self.language_classifier(language_emb)
        return language

