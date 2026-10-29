import flwr as fl
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

import os

# --------------------------------------------------
# Weight conversion helpers
# --------------------------------------------------

def get_parameters(model):
    return [
        val.cpu().numpy()
        for _, val in model.state_dict().items()
    ]


def set_parameters(model, parameters):
    params_dict = zip(
        model.state_dict().keys(),
        parameters
    )

    state_dict = {
        k: torch.tensor(v)
        for k, v in params_dict
    }

    model.load_state_dict(
        state_dict,
        strict=True
    )

# --------------------------------------------------
# Training
# --------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, cid=None):
    model.train()
    running_loss = 0.0

    pbar = tqdm(
        loader,
        leave=False,
        desc=f"Client {cid} Train"
    )

    for x, y in pbar:
        x = x.to(device)
        y = y.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_model(model,loader,criterion,device,cid=None,):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(
        loader,
        leave=False,
        desc=f"Client {cid} Eval"
    )

    with torch.no_grad():
        for x, y in pbar:
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device)

            logits = model(x)

            loss = criterion(logits,y)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct += (preds == y).sum().item()
            total += y.size(0)

            total_loss += loss.item()

    loss = total_loss / len(loader)
    acc = correct / total

    return loss, acc

def predictive_entropy(p):
    eps = 1e-8
    return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))

def enable_mc_dropout(model):
    """
    Enable dropout layers during test-time.
    Keep BatchNorm layers in eval mode.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def estimate_client_uncertainty(model, loader, device, mc_samples=50):
    model.eval()
    enable_mc_dropout(model)

    entropy_values = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            preds = []

            for _ in range(mc_samples):
                logits = model(x)
                probs = torch.sigmoid(logits)
                preds.append(probs)

            preds = torch.stack(preds)
            mean_prob = preds.mean(0)
            entropy = predictive_entropy(mean_prob)

            #entropy_values.extend(entropy.squeeze().cpu().numpy().tolist())
            entropy_values.append(entropy.mean().item())

    return float(np.mean(entropy_values))

# --------------------------------------------------
# Flower Client
# --------------------------------------------------

class FlowerClient(fl.client.NumPyClient):

    def __init__(self,model,trainloader,valloader,testloader,cid="0",fid="0",local_epochs=2, seed=42, dataSplit="211"):

        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.cid = cid
        self.fid = fid
        self.local_epochs = local_epochs
        self.best_val_loss = float("inf")

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(self.device)

        self.save_dir = "models/FedAvg-{}-seed{}/client-{}/checkpoint-{}".format(dataSplit, seed, cid, fid)
        os.makedirs(self.save_dir, exist_ok=True)

    # ---------------------------------------------
    # Send weights to server
    # ---------------------------------------------
    def get_parameters(self, config):
        return get_parameters(self.model)


    # ---------------------------------------------
    # Local training
    # ---------------------------------------------
    def fit(self,parameters,config    ):
        set_parameters(self.model,parameters)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-3)

        for epoch in range(self.local_epochs):
            train_loss = train_epoch(self.model,self.trainloader,optimizer,criterion,self.device,self.cid)
            val_loss, val_acc = evaluate_model(self.model,self.valloader,criterion,self.device,self.cid)

            print(
                f"[Client {self.cid}] "
                f"Epoch {epoch + 1:02d} | "
                f"Train={train_loss:.4f} | "
                f"Val={val_loss:.4f} | "
                f"ValAcc={val_acc:.2%}"
            )

            # always save
            #current_round = config["server_round"]
            #torch.save(self.model.state_dict(), os.path.join(self.save_dir, "epoch-{}.pth".format(current_round)))

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best.pth"))

                print(
                    f"Client {self.cid}: "
                    f"saved best model "
                    f"(val_loss={val_loss:.4f})"
                )


            # calculate entropy
            mean_entropy = estimate_client_uncertainty(self.model, self.valloader, self.device)

            # restore model state
            self.model.eval()

        return (get_parameters(self.model),len(self.trainloader.dataset),{"val_loss": float(val_loss),"entropy": mean_entropy})

    # ---------------------------------------------
    # Local evaluation
    # ---------------------------------------------
    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss, accuracy = evaluate_model(self.model, self.valloader, criterion, self.device, self.cid)

        print(
            f"[Client {self.cid}] "
            f"Val Loss={loss:.4f} | "
            f"Val Acc={accuracy:.2%}"
        )

        return (
            float(loss),
            len(self.valloader.dataset),
            {
                "accuracy": float(accuracy)
            },
        )