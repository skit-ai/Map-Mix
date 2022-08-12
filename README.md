# LangID-LRE2017


## Results

|                 | **Methods**          | **Accuracy** | **Weighted F1** | **Cluster Accuracy** | **ECE** |
|-----------------|----------------------|--------------|-----------------|----------------------|---------|
| baselines       | wav2vec2             | 0.598        | 0.577           | 0.938                | 0.310   |
|                 | Hubert               | 0.605        | 0.590           | 0.931                | 0.311   |
|                 | XLSR                 |              |                 |                      |         |
|                 | VoxLingua Pretrained |              |                 |                      |         |
| mixup baselines | XLSR static          |              |                 |                      |         |
|                 | XLSR latent          |              |                 |                      |         |
|                 | XLSR within          |              |                 |                      |         |
|                 | XLSR across          |              |                 |                      |         |