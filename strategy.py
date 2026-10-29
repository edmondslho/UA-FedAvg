import numpy as np
import torch

from flwr.server.strategy import FedAvg

from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)


def set_parameters(model, parameters):
    params_dict = zip(
        model.state_dict().keys(),
        parameters,
    )

    state_dict = {
        k: torch.tensor(v)
        for k, v in params_dict
    }

    model.load_state_dict(
        state_dict,
        strict=True,
    )


class SaveBestFedAvg(FedAvg):
    def __init__(self, model, checkpoint_file, *args, **kwargs, ):
        super().__init__(*args, **kwargs)

        self.model = model
        self.checkpoint_file = checkpoint_file
        self.best_loss = float("inf")
        self.latest_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures,
            )
        )

        if aggregated_parameters is not None:
            self.latest_parameters = (
                aggregated_parameters
            )

        print("Running FedAvg")

        return (
            aggregated_parameters,
            metrics,
        )

    def aggregate_evaluate(self, server_round, results, failures, ):
        print(f"aggregate_evaluate() called, round={server_round}")

        aggregated_loss, metrics = (
            super().aggregate_evaluate(
                server_round,
                results,
                failures,
            )
        )

        if aggregated_loss is None:
            return (
                aggregated_loss,
                metrics,
            )

        print(
            f"\nRound {server_round} "
            f"Val Loss = "
            f"{aggregated_loss:.6f}"
        )

        if aggregated_loss < self.best_loss:
            self.best_loss = aggregated_loss
            ndarrays = parameters_to_ndarrays(self.latest_parameters)

            set_parameters(self.model, ndarrays)
            torch.save(self.model.state_dict(), self.checkpoint_file)

            print(
                f"Saved best global model "
                f"(loss={aggregated_loss:.6f})"
            )

        return (aggregated_loss, metrics)


class UAFedAvg(FedAvg):

    def __init__(self, model, checkpoint_file, patience=15, withVLoss=False, *args, **kwargs, ):
        super().__init__(*args, **kwargs)

        self.model = model
        self.checkpoint_file = checkpoint_file
        self.best_val_loss = np.inf

        self.latest_parameters = None
        self.patience = patience
        self.counter = 0

        self.best_round = 0

        self.withVLoss = withVLoss
        self.alpha = 0.25

    # --------------------------------------------------
    # Convert Flower parameters -> PyTorch model
    # --------------------------------------------------

    def _set_model_parameters(self, model, parameters, ):

        params_dict = zip(model.state_dict().keys(), parameters)

        state_dict = {
            k: torch.tensor(v)
            for k, v in params_dict
        }

        model.load_state_dict(state_dict, strict=True)

    # --------------------------------------------------
    # Entropy-aware aggregation
    # --------------------------------------------------

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        #####################################
        # Compute uncertainty-aware weights
        ######################################
        client_weights = []

        for _, fit_res in results:
            entropy = fit_res.metrics.get("entropy", 1.0)
            vLoss = fit_res.metrics.get("val_loss", 1.0)
            num_examples = fit_res.num_examples

            if self.withVLoss:
                print("Running UA-FedAvg w/ vLoss")
                weight = (num_examples / (
                        1.0 + self.alpha * (vLoss + entropy)))  # UA-FedAvg w/ vLoss
            else:
                print("Running UA-FedAvg")
                weight = (num_examples / (entropy + 1e-8))  # UA-FedAvg

            client_weights.append(weight)

        total_weight = sum(client_weights)

        client_weights = [
            w / total_weight
            for w in client_weights
        ]

        ################################
        # Aggregate parameters manually
        ###############################

        client_ndarrays = []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            client_ndarrays.append(ndarrays)

        aggregated_ndarrays = []

        num_layers = len(client_ndarrays[0])

        for layer_idx in range(num_layers):
            layer_sum = None
            for weight, ndarrays in zip(client_weights, client_ndarrays):
                contribution = (weight * ndarrays[layer_idx])

                if layer_sum is None:
                    layer_sum = contribution
                else:
                    layer_sum += contribution

            aggregated_ndarrays.append(layer_sum)

        aggregated_parameters = (ndarrays_to_parameters(aggregated_ndarrays))
        self.latest_parameters = (aggregated_parameters)

        #################
        # Useful logging
        ################

        print(f"\n[ROUND {server_round}]")

        for idx, ((_, fit_res), weight) in enumerate(zip(results, client_weights)):
            print(
                f"Client {idx}: "
                f"entropy="
                f"val_loss={vLoss:.4f}, "
                f"{fit_res.metrics['entropy']:.4f}, "
                f"agg_weight={weight:.4f}"
            )

        return (aggregated_parameters, {})

    # --------------------------------------------------
    # Aggregate validation losses
    # Save best global model
    # --------------------------------------------------

    def aggregate_evaluate(
            self,
            server_round,
            results,
            failures,
    ):

        aggregated_loss, metrics = (
            super().aggregate_evaluate(
                server_round,
                results,
                failures,
            )
        )

        if aggregated_loss is None:
            return (aggregated_loss, metrics,)

        print(
            f"[ROUND {server_round}] "
            f"Global Val Loss="
            f"{aggregated_loss:.6f}"
        )

        #
        # Save best global model
        #
        if (aggregated_loss < self.best_val_loss):
            self.best_val_loss = (aggregated_loss)

            self.best_round = (server_round)

            self.counter = 0

            if self.latest_parameters is None:
                print(
                    "WARNING: no aggregated "
                    "parameters available yet"
                )

                return (aggregated_loss, metrics)

            ndarrays = parameters_to_ndarrays(self.latest_parameters)

            self._set_model_parameters(self.model, ndarrays)

            torch.save(self.model.state_dict(), self.checkpoint_file)

            print(
                f"*** New Best Model ***\n"
                f"Round: {server_round}\n"
                f"Val Loss: "
                f"{aggregated_loss:.6f}\n"
                f"Saved: "
                f"{self.checkpoint_file}"
            )
        else:
            self.counter += 1
            print(
                f"No improvement "
                f"({self.counter}/"
                f"{self.patience})"
            )
        return (aggregated_loss, metrics)
