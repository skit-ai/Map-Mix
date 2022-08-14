# LangID-LRE2017


## Results

|                 | **Methods**          | **Accuracy** | **Weighted F1** | **Cluster Accuracy** | **ECE** |
|-----------------|----------------------|--------------|-----------------|----------------------|---------|
| baselines       | wav2vec2             | 0.585        | 0.555           | 0.927                | 0.342   |
|                 | Hubert               | 0.596        | 0.571           | 0.932                | 0.326   |
|                 | XLSR                 |              |                 |                      |         |
|                 | VoxLingua Pretrained |              |                 |                      |         |
| mixup baselines | XLSR static          |              |                 |                      |         |
|                 | XLSR latent          |              |                 |                      |         |
|                 | XLSR within          |              |                 |                      |         |
|                 | XLSR across          |              |                 |                      |         |