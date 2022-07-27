import torch
import torch.nn as nn
import torch.nn.functional as F

import s3prl.hub as hub

# XLSR Mixup
class UpstreamTransformerXLSR(nn.Module):
    def __init__(self, upstream_model='xlsr_300m', feature_dim=1024, unfreeze_last_conv_layers=False):
        super().__init__()

        self.xlsrModelsUrls = {'xlsr_300m': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt',
        'xlsr_1b': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_960m_1000k.pt',
        'xlsr_2b': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_2B_1000k.pt'}

        self.upstream = getattr(hub, 'wav2vec2_url', self.xlsrModelsUrls[upstream_model])
        
        for param in self.upstream.parameters():
            param.requires_grad = False
        
        for param in self.upstream.model.encoder.layers.parameters():
            param.requires_grad = True

        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[5:].parameters():
                param.requires_grad = True
        
        self.language_classifier = nn.Sequential(
            nn.Linear(feature_dim, 14),
        )

    def simple_forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        x = torch.mean(x, dim=1)
        language = self.language_classifier(x)
        return language
    
    def mixup_forward(self, x, mixup_x, x_len, mixup_x_len, lam):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        mixup_x = [torch.narrow(mixup_wav,0,0,mixup_x_len[i]) for (i,mixup_wav) in enumerate(mixup_x.squeeze(1))]
        
        x = self.upstream(x)['last_hidden_state']
        mixup_x = self.upstream(mixup_x)['last_hidden_state']
        
        x = torch.mean(x, dim=1)
        mixup_x = torch.mean(mixup_x, dim=1)
        
        x = lam*x + (1-lam)*mixup_x

        language = self.language_classifier(x)
        return language

    def forward(self, x, mixup_x, x_len, mixup_x_len, apply_mixup, lam, mixup_type):
        if(apply_mixup):
            if mixup_type == 'latent-mixup-last':
                return self.mixup_forward(x, mixup_x, x_len, mixup_x_len, lam)
            elif mixup_type == 'static':
                x = lam*x + (1-lam)*mixup_x
                return self.simple_forward(x, x_len)
            else:
                raise Exception("Wrong mixup type. Supported types - 1. latent-mixup-last 2. static")
        else:
            return self.simple_forward(x, x_len)


