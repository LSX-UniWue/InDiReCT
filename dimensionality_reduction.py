import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import ortho_group


class NormalizedSoftmax():
    """
    Uses the NormalizedSoftmax loss function to find a good transformation.
    Is used for the Oracle baseline.
    """
    def __init__(self,
        num_components:int=2,
        patience:int=100,
        learning_rate:float=0.01,
    ) -> None:
        self.num_components = num_components

        self.U = None

        self.patience = patience
        self.lr = learning_rate

        self.loss_func = torch.nn.CrossEntropyLoss()

    def _loss(self, X:torch.Tensor, labels:torch.Tensor, U:torch.Tensor, P:torch.Tensor) -> torch.Tensor:
        # encode
        V = X @ U.T
        V = V / V.norm(dim=1, keepdim=True)

        # decode
        P_norm = P / P.norm(dim=1, keepdim=True)
        X_hat = V @ P_norm.T

        # loss
        loss = self.loss_func(X_hat, labels)

        return loss


    def fit(self, X:torch.Tensor, labels:torch.Tensor) -> None:
        # convert to tensor
        X = torch.as_tensor(X, dtype=torch.float)
        X = X / X.norm(dim=1, keepdim=True)

        num_classes = len(set(labels.numpy()))
        print(f"Fitting to {num_classes} classes")

        best_loss = torch.inf
        best_epoch = 0
        iteration = 0

        U = torch.normal(0, 0.1, (self.num_components, X.shape[1]), requires_grad=True, dtype=torch.float)
        P = torch.normal(0, 0.1, (num_classes, self.num_components), requires_grad=True, dtype=torch.float)

        optimizer = Adam([U, P], lr=self.lr)

        pbar = tqdm()
        while True:
            pbar.update(1)
            iteration += 1
            optimizer.zero_grad()
            
            loss = self._loss(X, labels, U, P)
            pbar.set_description(f"Loss ({iteration}): {loss.item():.4f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = iteration

                self.U = U.clone().detach()

            if iteration - best_epoch >= self.patience:
                break

            loss.backward()
            optimizer.step()

        print(f"Finished optimization. Best loss ({best_loss}) achieved after {best_epoch} iterations.")

    def transform(self, X:torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float)
            X = X / X.norm(dim=1, keepdim=True)

            V = X @ self.U.T
            V = V / V.norm(dim=1, keepdim=True)

            return V

    def fit_transform(self, X: torch.Tensor, labels:torch.Tensor=None) -> torch.Tensor:
        self.fit(X, labels)
        return self.transform(X)



class DimRedRecon():
    def __init__(self,
        num_components:int=2,
        patience:int=100,
        learning_rate:float=0.01
    ) -> None:
        self.num_components = num_components

        self.U = None

        self.lamb = 0
        self.patience = patience
        self.lr = learning_rate

    # from https://github.com/pytorch/pytorch/issues/8069#issuecomment-524096597
    # avoids gradient instabilities in acos at the edges
    def acos_safe(self, x, eps=1e-7):
        slope = np.arccos(1-eps) / eps
        buf = torch.empty_like(x)
        good = abs(x) <= 1-eps
        bad = ~good
        sign = torch.sign(x[bad])
        buf[good] = torch.acos(x[good])
        buf[bad] = torch.acos(sign * (1 - eps)) - slope*sign*(abs(x[bad]) - 1 + eps)
        return buf

    def _loss(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        # X = X / X.norm(dim=1, keepdim=True)

        # encode
        V = X @ U.T
        V = V / V.norm(dim=1, keepdim=True)

        # decode
        X_recon = V @ U
        X_recon = X_recon / X_recon.norm(dim=1, keepdim=True)

        # loss
        loss = torch.mean(self.acos_safe(torch.sum(X * X_recon, dim=1))) \
               + self.lamb * torch.sum(torch.square(U @ U.T - torch.eye(self.num_components)))

        return loss
        
    def fit(self, X:torch.tensor) -> None:
        # convert to tensor
        X = torch.as_tensor(X, dtype=torch.float)
        X = X / X.norm(dim=1, keepdim=True)

        best_loss = torch.inf
        best_epoch = 0
        epoch = 0

        U = torch.normal(0, 0.1, (self.num_components, X.shape[1]), requires_grad=True, dtype=torch.float)

        optimizer = Adam([U], lr=self.lr)

        pbar = tqdm()
        while True:
            epoch += 1
            pbar.update(1)
            optimizer.zero_grad()

            loss = self._loss(X, U)
            pbar.set_description(f"Loss ({epoch}): {loss.item()}")

            if loss.item() < best_loss:
                best_loss = loss
                best_epoch = epoch

                self.U = U.clone().detach()

            if epoch - best_epoch >= self.patience:
                break

            loss.backward()
            optimizer.step()

        print(f"Finished optimization. Best loss ({best_loss}) achieved after {best_epoch} iterations.")

    def transform(self, X:torch.tensor) -> np.array:
        # with torch.no_grad():
        X = torch.as_tensor(X, dtype=torch.float)
        X = X / X.norm(dim=1, keepdim=True)

        V = X @ self.U.T
        V = V / V.norm(dim=1, keepdim=True)

        return V

    def fit_transform(self, X: torch.tensor) -> np.array:
        self.fit(X)
        return self.transform(X)


class LinearAutoencoder():
    def __init__(self,
        num_components:int=2,
        patience:int=100,
        learning_rate:float=0.01
    ) -> None:
        self.num_components = num_components

        self.patience = patience
        self.lr = learning_rate

        self.encoder = None
        self.decoder = None
        
    def fit(self, X:torch.tensor) -> None:
        # convert to tensor
        X = torch.as_tensor(X, dtype=torch.float)

        best_loss = torch.inf
        best_epoch = 0
        epoch = 0

        encoder = torch.nn.Linear(X.shape[1], self.num_components)
        decoder = torch.nn.Linear(self.num_components, X.shape[1])

        optimizer = Adam(list(encoder.parameters())+list(decoder.parameters()), lr=self.lr)

        pbar = tqdm()
        while True:
            epoch += 1
            pbar.update(1)
            optimizer.zero_grad()

            # encode
            V = encoder(X)
            # decode
            X_recon = decoder(V)

            # loss
            loss = torch.sum(torch.square(X - X_recon))
            pbar.set_description(f"Loss ({epoch}): {loss.item()}")

            if loss.item() < best_loss:
                best_loss = loss
                best_epoch = epoch

                self.encoder = deepcopy(encoder)
                self.decoder = deepcopy(decoder)

            if epoch - best_epoch >= self.patience:
                break

            loss.backward()
            optimizer.step()

        print(f"Finished optimization. Best loss ({best_loss}) achieved after {best_epoch} iterations.")

    def transform(self, X:torch.tensor) -> np.array:
        X = torch.as_tensor(X, dtype=torch.float)
        with torch.no_grad():
            V = self.encoder(X)
        return V.detach()

    def fit_transform(self, X: torch.tensor) -> np.array:
        self.fit(X)
        return self.transform(X)


class Autoencoder():
    def __init__(self,
        num_components:int=2,
        patience:int=100,
        learning_rate:float=0.01
    ) -> None:
        self.num_components = num_components

        self.patience = patience
        self.lr = learning_rate

        self.encoder = None
        self.decoder = None

        self.hidden_size = 512 # best trade-off between speed and performance
        
    def fit(self, X:torch.tensor) -> None:
        # convert to tensor
        X = torch.as_tensor(X, dtype=torch.float)

        best_loss = torch.inf
        best_epoch = 0
        epoch = 0

        encoder = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], self.hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidden_size, self.num_components),
        )
        decoder = torch.nn.Sequential(
            torch.nn.Linear(self.num_components, self.hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidden_size, X.shape[1]),
        )

        optimizer = Adam(list(encoder.parameters())+list(decoder.parameters()), lr=self.lr, weight_decay=1e-2)

        pbar = tqdm()
        while True:
            epoch += 1
            pbar.update(1)
            optimizer.zero_grad()

            # encode
            V = encoder(X)
            # decode
            X_recon = decoder(V)

            # loss
            # loss = torch.mean(torch.sqrt(torch.sum(torch.square(X - X_recon), dim=1)))
            loss = torch.sum(torch.square(X - X_recon)) # Loss from LAE
            pbar.set_description(f"Loss ({epoch}): {loss.item()}")

            if loss.item() < best_loss:
                best_loss = loss
                best_epoch = epoch

                self.encoder = deepcopy(encoder)
                self.decoder = deepcopy(decoder)

            if epoch - best_epoch >= self.patience:
                break

            loss.backward()
            optimizer.step()

        print(f"Finished optimization. Best loss ({best_loss}) achieved after {best_epoch} iterations.")

    def transform(self, X:torch.tensor) -> np.array:
        X = torch.as_tensor(X, dtype=torch.float)
        with torch.no_grad():
            V = self.encoder(X)
        return V.detach()

    def fit_transform(self, X: torch.tensor) -> np.array:
        self.fit(X)
        return self.transform(X)

