# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## Report
Wandb report: https://wandb.ai/hsee/pytorch_template_asr_example/reports/ASR-REPORT--Vmlldzo5NzA2MjAz
## About

This repository is my realisationf of ASR problem. To run train you should do several steps. First of all install requirements. 
```bash
pip3 install -r ./requirements.txt
```
To run your training should write following:
```bash
python3 train.py HYDRA_CONFIG_ARGUMENTS
```

My checkpoint path: https://disk.yandex.com/d/srKCqb83Em5kWw
To run inference (evaluate the model or save predictions) don't forget to change path to checkpoint:

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
