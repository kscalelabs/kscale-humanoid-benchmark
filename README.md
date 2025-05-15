<div align="center">
<h1>K-Sim Gym</h1>
<p>Train and deploy your own humanoid robot controller in 700 lines of Python</p>
<h3>
  <a href="https://url.kscale.dev/leaderboard">Leaderboard</a> ·
  <a href="https://url.kscale.dev/docs">Documentation</a> ·
  <a href="https://github.com/kscalelabs/ksim/tree/master/examples">K-Sim Examples</a>
</h3>

https://github.com/user-attachments/assets/3d44aa23-5ad7-41a3-b467-22165542b8c4

</div>

## Getting Started

You can use this repository as a GitHub template or as a Google Colab.

### Google Colab

You can quickly try out the humanoid benchmark by running the [training notebook](https://colab.research.google.com/github/kscalelabs/ksim-gym/blob/master/train.ipynb) in Google Colab.

### On your own GPU

1. Read through the [current leaderboard](https://url.kscale.dev/leaderboard) submissions and through the [ksim examples](https://github.com/kscalelabs/ksim/tree/master/examples)
2. Create a new repository from this template by clicking [here](https://github.com/new?template_name=ksim-gym&template_owner=kscalelabs)
3. Make sure you have installed `git-lfs`:

```bash
sudo apt install git-lfs  # Ubuntu
brew install git-lfs  # MacOS
```

4. Clone the new repository you create from this template:

```bash
git clone git@github.com:<YOUR USERNAME>/ksim-gym.git
cd ksim-gym
```

5. Create a new Python environment (we require Python 3.11 or later)
6. Install the package with its dependencies:

```bash
pip install -r requirements.txt
pip install 'jax[cuda12]'  # If using GPU machine, install Jax CUDA libraries
```

7. Train a policy:

```bash
python -m train
```

8. Convert the checkpoint to a `kinfer` model:

```bash
python -m convert /path/to/ckpt.bin /path/to/model.kinfer
```

9. Visualize the converted model:

```bash
kinfer-sim assets/model.kinfer kbot --save-video assets/video.mp4
```

10. Commit the K-Infer model and the recorded video to this repository
11. Push your code and model to your repository, and make sure the repository is public
12. Write a message with a link to your repository on our [Discord](https://url.kscale.dev/discord) in the "【🧠】submissions" channel
13. Wait for one of us to run it on the real robot - this should take about a day, but if we are dragging our feet, please message us on Discord
14. Voila! Your name will now appear on our [leaderboard](https://url.kscale.dev/leaderboard)

## Troubleshooting

If you encounter issues, please consult the [ksim documentation](https://docs.kscale.dev/docs/ksim#/) or reach out to us on [Discord](https://url.kscale.dev/docs).

## Tips and Tricks

To see all the available command line arguments, use the command:

```bash
python -m train --help
```

To visualize running your model without using `kos-sim`, use the command:

```bash
python -m train run_mode=view
```

This repository contains a pre-trained checkpoint of a model which has been learned to be robust to pushes, which is useful for both jump-starting model training and understanding the codebase. To initialize training from this checkpoint, use the command:

```bash
python -m train load_from_ckpt_path=assets/ckpt.bin
```

You can visualize the pre-trained model by combining these two commands:

```bash
python -m train load_from_ckpt_path=assets/ckpt.bin run_mode=view
```

If you want to use the Jupyter notebook and don't want to commit your training logs, we suggest using [pre-commit](https://pre-commit.com/) to clean the notebook before committing:

```bash
pip install pre-commit
pre-commit install
```
