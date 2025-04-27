<div align="center">
<h1>K-Scale Humanoid Benchmark</h1>
<p>So you think you have what it takes to train a reinforcement learning policy, huh? Now's your chance to prove it!</p>
<h3>
  <a href="https://url.kscale.dev/leaderboard">Leaderboard</a> ·
  <a href="https://url.kscale.dev/docs">Documentation</a> ·
  <a href="https://github.com/kscalelabs/ksim/tree/master/examples">K-Sim Examples</a>
</h3>
</div>

![K-Bot](/assets/banner.png)

## Getting Started

1. Read through the [current leaderboard](https://url.kscale.dev/leaderboard) submissions and through the [ksim examples](https://github.com/kscalelabs/ksim/tree/master/examples)
2. Create a new repository from this template by clicking [here](https://github.com/new?template_name=kscale-humanoid-benchmark&template_owner=kscalelabs)
3. Make sure you have installed `git-lfs`:

```bash
sudo apt install git-lfs  # Ubuntu
brew install git-lfs  # MacOS
```

4. Clone the resulting repository:

```bash
git clone git@github.com:<YOUR USERNAME>/kscale-humanoid-benchmark.git
cd kscale-humanoid-benchmar
```

5. Create a new Python environment (we require Python 3.11 or later)
6. Install the package with it's dependencies:

```bash
pip install .
```

7. Train a policy:

```bash
python -m benchmark.train
```

8. Update the policy weights in `assets` and run the deployment script on your new policy:

```bash
python -m benchmark.deploy
```

8. Add a video of your new policy to this README
9. Push your code and model to your repository, and make sure the repository is public
10. Write a message with a link to your repository on our [Discord](https://url.kscale.dev/discord) in the "【🧠】submissions" channel
11. Wait for one of us to run it on the real robot - this should take about a day
12. Viola! Your name will now appear on our [leaderboard](https://url.kscale.dev/leaderboard)

## Tips and Tricks

To see all the available command line arguments, use the command:

```bash
python -m benchmark.train --help
```

To visualize running your model without using `kos-sim`, use the command:

```bash
python -m benchmark.train run_model_viewer=True
```

This repository contains a pre-trained checkpoint, which is useful for both jump-starting model training and understanding the codebase. To initialize training from this checkpoint, use the command:

```bash
python -m benchmark.train load_from_ckpt_path=assets/ckpt.bin
```
