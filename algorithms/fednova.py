import torch
from utils.model import *
from utils.utils import *
from disco import *
from algorithms.client import local_train_net

def fednova_alg(args, n_comm_rounds, nets, global_model, party_list_rounds,
                net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts,
                device, global_dist, logger):

    # Print initial setup only once
    logger.info("---FEDNOVA---")
    print("---FEDNOVA---")

    # Initialize tracking variables
    best_test_acc = 0
    record_test_acc_list = []

    # Ensure global model is on CUDA
    global_model = global_model.cuda()

    # Print dataset information once
    logger.info(f"Length of testing set: {len(test_dl.dataset)}")
    print(f"Length of testing set: {len(test_dl.dataset)}")
    logger.info("Training begins!")
    print("Training begins!")

    for round in range(n_comm_rounds):
        logger.info(f"In communication round: {round}")
        print(f"In communication round: {round}")

        # Get participating clients for this round
        party_list_this_round = party_list_rounds[round]
        global_w = global_model.state_dict()

        # Select participating nets
        nets_this_round = {k: nets[k] for k in party_list_this_round}

        # Move networks to CUDA and load global weights
        for net in nets_this_round.values():
            net.cuda()
            net.load_state_dict(global_w)

        # Calculate initial aggregation weights
        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

        if round == 0 or args.sample_fraction < 1.0:
            print(f'Dataset size weight : {fed_avg_freqs}')

        # DISCO weight adjustment if enabled
        if args.disco:
            distribution_difference = get_distribution_difference(
                traindata_cls_counts,
                participation_clients=party_list_this_round,
                metric=args.measure_difference,
                hypo_distribution=global_dist
            )
            fed_avg_freqs = disco_weight_adjusting(
                fed_avg_freqs,
                distribution_difference,
                args.disco_a,
                args.disco_b
            )
            if round == 0 or args.sample_fraction < 1.0:
                print(f'Distribution_difference : {distribution_difference}')
                print(f'Disco Aggregation Weights : {fed_avg_freqs}')

        # Local training
        local_train_net(nets_this_round, args, net_dataidx_map,
                       train_dl=train_local_dls, test_dl=test_dl,
                       device=device, logger=logger)

        # Calculate FedNova coefficients
        client_step = [len(net_dataidx_map[r]) // args.batch_size for r in party_list_this_round]

        # Ensure client_step does not contain zero values
        for i in range(len(client_step)):
            if client_step[i] == 0:
                client_step[i] = 1  # Set to 1 to avoid division by zero

        tao_eff = 0.0
        for j in range(len(fed_avg_freqs)):
            tao_eff += fed_avg_freqs[j] * client_step[j]

        correct_term = sum(freq / step * tao_eff for freq, step in zip(fed_avg_freqs, client_step))

        # Initialize new global weights
        for key in global_w.keys():
            global_w[key] = (1.0 - correct_term) * global_w[key].cuda()

        # Aggregate models
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            contribution = fed_avg_freqs[net_id] / client_step[net_id] * tao_eff

            for key in net_para:
                global_w[key] += net_para[key].cuda() * contribution

        # Update global model
        global_model.load_state_dict(global_w)

        # Test global model
        test_acc, conf_matrix, _ = compute_accuracy(
            global_model, test_dl,
            get_confusion_matrix=True,
            device=device
        )

        record_test_acc_list.append(test_acc)

        # Update best accuracy
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            logger.info(f'New Best test acc: {test_acc:.4f}')

        logger.info(f'>> Global Model Test accuracy: {test_acc:.4f}')
        logger.info(f'>> Global Model Best accuracy: {best_test_acc:.4f}')
        print(f'>> Global Model Test accuracy: {test_acc:.4f}, Best: {best_test_acc:.4f}')

        # Save model if required
        if args.save_model:
            mkdirs(args.modeldir + 'fednova/')
            save_path = args.modeldir + 'fednova/globalmodel' + args.log_file_name + '.pth'
            torch.save(global_model.state_dict(), save_path)

    return record_test_acc_list, best_test_acc
