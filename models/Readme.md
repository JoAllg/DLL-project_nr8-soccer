For the training and result graphs on our poster we used an older checkpoint `v1_2-2zwe9huo.cleanrl_model` which was trained over a longer time without breaks or restarts in the middle (continous graphs are available).

Weights and Biases link:
https://wandb.ai/models-albert-ludwigs-universit-t-freiburg/joshua/runs/2zwe9huo?

We also hand in our currently best model for ~2vs0 stages (scoring goals): `v1_3-laafgmd0.cleanrl_model` [Weights&Biases](https://wandb.ai/models-albert-ludwigs-universit-t-freiburg/joshua/runs/laafgmd0?).

And the results for a curriculum trained on passing-rewards only: `passing_v3-okrui4ax.cleanrl_model` [Weights&Biases](https://wandb.ai/models-albert-ludwigs-universit-t-freiburg/joshua/runs/okrui4ax?).

The prefix of the checkpoint names denounce the config from configs/ it was trained on last. But this does not mean that it was trained on all stages in that configuration and the models were trained on top of checkpoints from previous configurations.