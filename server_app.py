from flwr.server import ServerApp, ServerAppComponents, ServerConfig
import os

from util import set_seed
from InfantMovementCNN import InfantMovementCNN
from strategy import SaveBestFedAvg, UAFedAvg


def create_model():
    return InfantMovementCNN()


def weighted_average(metrics):
    accuracies = [n * m["accuracy"] for n, m in metrics]
    examples = [n for n, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


def make_server_app(fold_id, seed=42, dataSplit="211", aggStrategy="UA-FedAvg", withVLoss=False):
    def server_fn(context):
        # create folder to save the checkpoints
        modelName = aggStrategy
        if aggStrategy == "UA-FedAvg":
            if withVLoss:
                modelName = "{}withVLoss".format(modelName)

        savePath = "models/{}-{}-seed{}/global/checkpoint-{}".format(modelName, dataSplit, seed, fold_id)
        os.makedirs(savePath, exist_ok=True)
        savePath = os.path.join(savePath, "best.pth")

        # fix the seed
        set_seed(seed)
        global_model = create_model()

        if aggStrategy == "FedAvg":
            strategy = SaveBestFedAvg(  # FedAvg
                model=global_model,
                checkpoint_file=savePath,

                fraction_fit=1.0,
                fraction_evaluate=1.0,

                min_fit_clients=3,
                min_evaluate_clients=3,
                min_available_clients=3,

                evaluate_metrics_aggregation_fn=
                weighted_average,
            )
        elif aggStrategy == "UA-FedAvg":
            strategy = UAFedAvg(  # UAFedAvg
                model=global_model,
                checkpoint_file=savePath,

                withVLoss=withVLoss,

                fraction_fit=1.0,
                fraction_evaluate=1.0,

                min_fit_clients=3,
                min_evaluate_clients=3,
                min_available_clients=3,

                evaluate_metrics_aggregation_fn=
                weighted_average,
            )

        config = ServerConfig(num_rounds=100)

        return ServerAppComponents(strategy=strategy, config=config)

    return ServerApp(server_fn=server_fn)
