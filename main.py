import os
import json
import copy
import torch
import random
import logging
import datetime
import numpy as np
from args import get_args
from disco import *
from algorithms import *
from utils.model import *
from utils.utils import *

if __name__ == '__main__':
    # Configurations
    args = get_args()

    # Create necessary directories
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    mkdirs(args.datadir)

    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    print(now_time)

    dataset_logdir = os.path.join(args.logdir, args.dataset)
    mkdirs(dataset_logdir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % (now_time)
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(dataset_logdir, argument_path), 'w') as f:
        json.dump(str(args), f)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (now_time)
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(dataset_logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')
    logger = logging.getLogger()

    # Modified device setup
    use_cuda = torch.cuda.is_available() and args.device.startswith('cuda')
    device = torch.device('cuda' if use_cuda else 'cpu')

    if use_cuda:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")

    seed = args.init_seed
    logger.info("#" * 100)

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    # Convert download_data to boolean for dataset loading
    args.download = bool(args.download_data)

    # Data partition of clients
    print("Training set partition")
    net_dataidx_map, traindata_cls_counts = partition_data(args)

    # Choose clients per round
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    # Get the testing dataloader
    imbalanced_test_dist = None
    if args.test_imb == 0: # Full testing set
        _, test_dl = get_dataloader(args)
    else:
        print("Testing set partition")
        split_test_dataidx, _, imbalanced_test_dist = partition_split_test_data(args)
        test_dl = get_dataloader_split_test(args, split_test_dataidx[0])
        imbalanced_test_dist = np.array(imbalanced_test_dist) / len(test_dl.dataset)

    global_dist = np.ones(traindata_cls_counts.shape[1])/traindata_cls_counts.shape[1] if args.test_imb == 0 else imbalanced_test_dist
    print("Length of testing set:", len(test_dl.dataset))

    # Modified model initialization to handle device properly
    nets = init_nets(args.n_parties, args, device=device)
    global_models = init_nets(1, args, device=device)
    global_model = global_models[0]

    n_comm_rounds = args.comm_round
    if args.load_model_file:
        state_dict = torch.load(args.load_model_file, map_location=device)
        global_model.load_state_dict(state_dict)
        n_comm_rounds -= args.load_model_round

    # Get the training dataloaders
    train_local_dls = []
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        train_dl_local, _ = get_dataloader(args, dataidxs)
        train_local_dls.append(train_dl_local)

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    else:
        moment_v = None

    acc_dir = os.path.join(dataset_logdir, 'acc_list')
    if not os.path.exists(acc_dir):
        os.mkdir(acc_dir)
    acc_path = os.path.join(dataset_logdir, f'acc_list/{now_time}.npy')

    # Training with proper error handling
    print("\nTraining begins!")
    try:
        if args.alg == 'moon':
            print("----MOON----\n")
            record_test_acc_list, best_test_acc = moon_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, device, global_dist, logger)
        elif args.alg == 'fedavg':
            print("----FEDAVG----\n")
            record_test_acc_list, best_test_acc = fedavg_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger)
        elif args.alg == 'scaffold':
            print("----SCAFFOLD----\n")
            record_test_acc_list, best_test_acc = scaffold_alg(args, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, device, global_dist, logger)
        elif args.alg == 'feddyn':
            print("----FEDDYN----\n")
            record_test_acc_list, best_test_acc = feddyn_alg(args, n_comm_rounds, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, device, global_dist, logger)
        elif args.alg == 'feddc':
            print("----FEDDC----")
            record_test_acc_list, best_test_acc = feddc_alg(args, n_comm_rounds, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, device, global_dist, logger)
        elif args.alg == 'fednova':
            print("---FEDNOVA---")
            record_test_acc_list, best_test_acc = fednova_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, device, global_dist, logger)
        elif args.alg == 'fedprox':
            print("---FEDPROX---\n")
            record_test_acc_list, best_test_acc = fedprox_alg(args, n_comm_rounds, nets, global_model, party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts, moment_v, device, global_dist, logger)
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

    np.save(acc_path, np.array(record_test_acc_list))
    print('>> Global Model Best accuracy: %f' % best_test_acc)
    print(args)
    print(now_time)
    print('>> Done!')
