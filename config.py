from benchmark.scenario.registery import scenario_register


num_classes = scenario_register('Image Classification (32*32)', dict(
    source_datasets_name=['CIFAR10', 'SVHN'],
    target_datasets_order=['STL10', 'MNIST', 'STL10', 'USPS', 'MNIST', 'STL10'],
    da_mode='da',
    num_samples_in_each_target_domain=100,
    data_dirs={d: f'/data/zql/datasets/{d}' for d in ['CIFAR10', 'SVHN', 'USPS', 'MNIST', 'STL10']}
))

pretrained_model_path = './example/exp/offline/logs/1664372604.1563218/models/main model.pt'
