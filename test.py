import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances   ###
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['root_dir'],
        mode='test',
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        num_workers=2
    )
    ###
    
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    class_num = 11
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    classwise_num = {}
    classwise_correct = {}
    classwise_accuracy = {}
    # init dict to 0s.
    for i in range(11):
        classwise_num[i] = 0
        classwise_correct[i] = 0
        classwise_accuracy[i] = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

            ## classwise accuracy
            pred = torch.argmax(output, dim=1)
            target_index = torch.argmax(target, dim=1)
            for i in range(len(target_index)):
                classwise_num[target_index[i].item()] += 1
                if pred[i] == target_index[i]:
                    classwise_correct[target_index[i].item()] += 1


    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    ## print classwise accuracy
    print(f'classwise_num: {classwise_num}')
    print(f'classwise_correct: {classwise_correct}')
    for key in classwise_num.keys():
        classwise_accuracy[key] = classwise_correct[key] / classwise_num[key]
    print(f'classwise_accuracy: {classwise_accuracy}')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
