<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->
<!-- /code_chunk_output -->

# Fine-grained Image Classification of Korean Food

## Introduction
Building a simple neural network to classify the 11 detailed classes of Korean Kimchi

Classes = ['갓김치', '깍두기', '나박김치', '무생채', '배추김치', '백김치', '부추김치', '열무김치', '오이소박이', '총각김치', '파김치']

## Dataset
### Download the dataset
1. Register you in [NIA's dataset lake](https://aihub.or.kr/join/mberSe.do?currMenu=108&topMenu=108)
2. Go to [dataset page in NIA for this project (Korean food dataset)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=79) to download the dataset.
    - Note that the entire dataset is 15.73GB but we are going to use a portion of it.

### Prepare the dataset
1. Copy the dataset to the code directory
1. Run the following command in a shell (`cmd` or `console` in Windows -- We only test this in Unix like OS such as Mac OS)
```
$ python prepare_dataset.py
```

## Train
```
$ python train.py -c config.json
```
```
$ python train.py -r path/to/ckpt_file.pth
```

## Test
```
$ python test.py -r path/to/ckpt_file.pth
```

## Tensorboard Visualization
```
$ tensorboard --logdir saved/log/
```

## Folder Structure
  ```
  kimchi_classification/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```
