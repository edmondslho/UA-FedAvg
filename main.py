######################################################################
# Official implmentation of UA-FedAvg
#
# "Uncertainty-Aware Federated Learning for Infant Movement Analysis"
# Proceedings of FLTA 2026
# Pre-print: arXiv:2609.31463
#
# Author: Edmond S. L. Ho (Shu-Lim.Ho@glasgow.ac.uk)
# GitHub: https://github.com/edmondslho/UA-FedAvg
# Version: 1.0
# Last Updated: 28 Oct 2026
######################################################################

from flwr.simulation import run_simulation

from client_app import make_client_app
from server_app import make_server_app

seed = 42  # fixing random seed
dataSplit = "111-211"  # selecting train-test ratio among the 3 clients
aggStrategy = "UA-FedAvg"  # to select between FedAvg and UA-FedAvg
withVLoss = True  # to select between UA-FedAvg with and without validation loss

for fold_id in range(1, 10):  # 9-fold cross-validation - following Kulvicius et al., Communications medicine, 2025
    client_app = make_client_app(fold_id=fold_id, seed=seed, dataSplit=dataSplit)
    server_app = make_server_app(fold_id=fold_id, seed=seed, dataSplit=dataSplit, aggStrategy=aggStrategy,
                                 withVLoss=withVLoss)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=3,  # 3 clients in this simulation
    )
