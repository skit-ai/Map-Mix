# Map-Mix


## Results

|                 | **Methods**          | **Accuracy** | **Weighted F1** | **Cluster Accuracy** | **ECE** |
|-----------------|----------------------|--------------|-----------------|----------------------|---------|
| baselines       | wav2vec2             | 0.598        | 0.577           | 0.938                | 0.31    |
|                 | Hubert               | 0.605        | 0.59            | 0.931                | 0.311   |
|                 | XLSR                 | 0.64         | 0.605           | 0.95                 | 0.282   |
|                 | VoxLingua Pretrained |              |                 |                      |         |
| mixup baselines | XLSR static          |              |                 |                      |         |
|                 | XLSR latent          |              |                 |                      |         |
|                 | XLSR within          |              |                 |                      |         |
|                 | XLSR across          |              |                 |                      |         |
