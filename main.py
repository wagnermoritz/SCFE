import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

import os
import time
import pandas as pd
import sklearn
import argparse
import sklearn.linear_model
from sklearn.decomposition import PCA

from Models import *
from SCFE import *

# for the experiments in the paper, lam0 = 1.0 and L0 = 0.01
# the values for beta and theta were found using a grid search
# KDE (0-norm, 1-norm), GMM (0-norm, 1-norm), kNN (0-norm, 1-norm)
# DNN experiments:
# Wine data set:
# beta: (2, 15), (2, 5), (2, 30)
# theta: (75, 75), (0.5, 0.75), (0.5, 0.5)
# Housing data set:
# beta: (2, 10), (2, 7.5), (2, 5)
# theta: (25, 20), (0.5, 0.5), (1, 0.75)
# MNIST data set: (no KDE)
# beta: (25, 15), (25, 10)
# theta: (15, 10), (15, 10)
# Linear classifier experiments:
# Wine data set:
# beta: (1, 5), (1, 50), (1, 10)
# theta: (1, 1), (0.5, 0.5), (1, 1)
# Housing data set:
# beta: (1, 5), (1, 10), (1, 10)
# theta: (2, 1.5), (0.25, 0.25), (1, 1)

class CDataset(Dataset):
  def __init__(self, x, y):

    self.x_train = x.clone()
    self.y_train = y.clone()

  def __len__(self):
    return len(self.y_train)

  def __getitem__(self,idx):
    return self.x_train[idx], self.y_train[idx]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', choices=['Housing', 'Wine', 'Wisconsin', 'MNIST'], type=str, required=True)
    parser.add_argument('--alg', choices=['SCFE_KDE', 'SCFE_GMM', 'SCFE_kNN'],
                        type=str, required=True, help='The SCFE variant to be tested.')
    parser.add_argument('--stepsize', type=float, required=True, help='The stepsize for the SCFE.')
    parser.add_argument('--sparsity', type=float, required=True, help='The sparsity parameter for the SCFE.')
    parser.add_argument('--plausibility', type=float, required=True, help='The plausibility parameter for the SCFE.')
    parser.add_argument('--n_neighbors', type=int, required=True, help='The number of neighbors for the SCFE_kNN.')
    parser.add_argument('--prox', choices=['zero', 'half', 'one', 'clamp', 'zero_fixed'], type=str,
                        required=True, help='The prox operator for the SCFE.')
    parser.add_argument('--model', choices=['Linear', 'DNN'], type=str, required=True, help='The model to be used.')
    parser.add_argument('--resdir', type=str, required=True, help='Directory for the output files.')
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(device)

    os.makedirs(args.resdir, exist_ok=True)

    if not args.dataset == 'MNIST':
        if args.dataset == 'Wisconsin':
            DATASET = 'wdbc.csv'
            FEATURE_COLS = ['radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 
                            'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1', 'radius2',
                            'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 'concavity2',
                            'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 'texture3',
                            'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3',
                            'concave_points3', 'symmetry3', 'fractal_dimension3']
            TARGET_COL = 'Diagnosis'
            NUMNEUR = 50
            REPLACE = [['M', '1'], ['B', '0']]
        if args.dataset == 'Housing':
            DATASET = 'BostonHousing.csv'
            FEATURE_COLS = ["crim","zn","indus","nox","rm","age","dis","rad","tax","ptratio","b","lstat"]
            TARGET_COL = 'target'
            NUMNEUR = 20
            REPLACE = [['N', '1'], ['P', '0']]
        if args.dataset == 'Wine':
            DATASET = 'wine.csv'
            FEATURE_COLS = ['alc', 'mal_acid', 'ash', 'alc_ash', 'mg', 'phen', 'flav', 'nonflav',
                            'proa', 'color', 'hue', 'dil', 'prol']
            TARGET_COL = 'class'
            NUMNEUR = 20
            REPLACE = [['N', '1'], ['P', '0']]

        numclasses = 2
        testlen = 100
        epochs = 20
        lr = 0.01
        batch_size=32
        shuffle=False
        lossfn = nn.BCELoss()
        pca_components = None if args.model == 'DNN' else 8
        
        # Load and preprocess the data
        df = pd.read_csv('./datasets/' + DATASET)
        df = df.dropna()
        df = df.replace(*REPLACE[0])
        df = df.replace(*REPLACE[1])
        df = df[FEATURE_COLS + [TARGET_COL]].astype("float32")
        df.loc[:, df.columns != TARGET_COL] = ((df.loc[:, df.columns != TARGET_COL]
                                            - df.loc[:, df.columns != TARGET_COL].mean()) 
                                            / df.loc[:, df.columns != TARGET_COL].std())
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df_train = df[:len(df) - testlen]
        df_test = df[len(df) - testlen:]

        x_test = torch.tensor(df_test.loc[:, df.columns != TARGET_COL].values, dtype=torch.float32)
        y_test = torch.tensor(df_test.loc[:, df.columns == TARGET_COL].values, dtype=torch.float32)
        x_train = torch.tensor(df_train.loc[:, df.columns != TARGET_COL].values, dtype=torch.float32)
        y_train = torch.tensor(df_train.loc[:, df.columns == TARGET_COL].values, dtype=torch.float32)

        # PCA if we are using a linear model
        if pca_components is not None:
            pca = PCA(n_components=pca_components)
            pca.fit(x_train.numpy())
            pmat = torch.from_numpy(pca.components_).type(torch.float32)
            pmean = torch.from_numpy(pca.mean_).type(torch.float32).view(1, -1)
            x_train = (x_train - pmean) @ pmat.T
            x_test = (x_test - pmean) @ pmat.T

        # for min-max scaling in the SCFE class
        mins = torch.cat([x_test, x_train]).min(dim=0)[0]
        maxs = torch.cat([x_test, x_train]).max(dim=0)[0]

        # train the model
        dataset = CDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        if args.model == 'DNN':
            model = nn.Sequential(nn.Linear(len(FEATURE_COLS) if pca_components is None else pca_components, NUMNEUR),
                                nn.ReLU(),
                                nn.Linear(NUMNEUR, NUMNEUR),
                                nn.ReLU(),
                                nn.Linear(NUMNEUR, 1),
                                nn.Sigmoid())
            _ = train(model, dataloader, epochs=epochs, lr=lr, loss_fn=lossfn)
            model.eval()
            model.to(device)
            print('Accuracy:', ((model(x_test.to(device)) >= 0.5) == y_test.to(device)).float().mean())
        else:
            model = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
            model.fit(x_train, y_train)
            model = LinearModel(torch.from_numpy(model.coef_), torch.from_numpy(model.intercept_))
            model.to(device=device, dtype=x_test.dtype)
            print('Accuracy:', (model(x_test.to(device)) * y_test.to(device) > 0).float().mean())

        # train the KDEs and GMMs
        kdes = train_n_kdes(x_train, y_train[:, 0], numclasses=2)
        gmms = train_n_GMMs(x_train, y_train[:, 0], numclasses=2)
        gmms[0].to(device=device)
        gmms[1].to(device=device)
        kdes[0].to(device=device)
        kdes[1].to(device=device)

        # train the LOFs
        LOF1 = sklearn.neighbors.LocalOutlierFactor(n_neighbors=30, novelty=True)
        LOF1.fit(x_train[y_train.view(-1)==1])
        LOF0 = sklearn.neighbors.LocalOutlierFactor(n_neighbors=30, novelty=True)
        LOF0.fit(x_train[y_train.view(-1)==0])

    else:
        # Load and preprocess the MNIST data
        b_size = 128
        numclasses = 10
        normalize = transforms.Normalize([.5], [.5])
        test_set = torchvision.datasets.MNIST("./Saves/Data/", download=True, train=False,
                                                transform=transforms.Compose([transforms.ToTensor(), normalize]))
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=b_size, shuffle=False, drop_last=False)

        train_set = torchvision.datasets.MNIST("./Saves/Data/", download=True, train=True,
                                                transform=transforms.Compose([transforms.ToTensor(), normalize]))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=b_size, drop_last=False)

        # Create tensors from the training and test data
        x_train, y_train = [], []
        counts = [0 for _ in range(10)]
        for point in train_set:
            if counts[point[1]] < 2000:
                x_train.append(point[0])
                y_train.append(point[1])
                counts[point[1]] += 1
        x_train = torch.stack(x_train, dim=0)
        y_train = torch.tensor(y_train).view(-1, 1)

        x_test, y_test = [], []
        counts = [0 for _ in range(10)]
        for point in test_set:
            if counts[point[1]] < 100:
                x_test.append(point[0])
                y_test.append(point[1])
                counts[point[1]] += 1
        x_test = torch.stack(x_test, dim=0).view(len(x_test), -1)
        y_test = torch.tensor(y_test).view(-1, 1)

        # for min-max scaling in the SCFE class
        maxs = x_test.max().repeat(x_test.size(-1))
        mins = x_test.min().repeat(x_test.size(-1))

        # train the model
        model = MNISTCNN()
        train(model, train_loader, epochs=80)
        model.eval()
        model.to(device)

        n, succ = 0, 0
        for x, y in dataloader:
            y = y.to(device)
            out = model(x.to(device))
            pred = out.argmax(dim=1)
            n += x.shape[0]
            succ += (pred == y).float().sum().item()
        print(f'Accuracy: {succ/n}')

        # train the GMMs
        gmms = train_n_GMMs(x_train.to(device).view(x_train.size(0), -1), y_train.to(device).view(-1), numclasses=10, K=10)
        for i in range(len(gmms)):
            gmms[i].to(device=device)
        # train the LOFs
        LOFs = [sklearn.neighbors.LocalOutlierFactor(n_neighbors=50, novelty=True) for _ in range(10)]
        for i in range(10):
            LOFs[i].fit(x_train.view(x_train.size(0), -1)[y_train.view(-1) == i])

    target = (-2 * y_test + 1).long()
    if args.alg == 'SCFE_KDE':
        Explainer = APG0_CFE_KDE(model, mins, maxs, numclasses=numclasses, kdes=gmms, range_min=None, range_max=None,
                                 lam0=1.0, c=0.0, beta=args.sparsity, theta=args.plausibility, iters=200, L0=0.01,
                                 linesearch=False, prox=args.prox)
    elif args.alg == 'SCFE_GMM':
        Explainer = APG0_CFE_KDE(model, mins, maxs, numclasses=numclasses, kdes=gmms, range_min=None, range_max=None,
                                 lam0=1.0, c=0.0, beta=args.sparsity, theta=args.plausibility, iters=200, L0=0.01,
                                 linesearch=False, prox=args.prox)
    elif args.alg == 'SCFE_kNN':
        target = (y_test + 1) % 10
        Explainer = APG0_CFE_kNN(model, mins, maxs, numclasses=numclasses,
                                 trains=[x_train.view(x_train.size(0), -1)[y_train.view(-1) == i] for i in range(numclasses)],
                                 range_min=None, range_max=None, lam0=1.0, c=0.0, beta=args.sparsity, theta=args.plausibility,
                                 iters=200, L0=0.01, k=args.n_neighbors, prox=args.prox, linesearch=False)

    # Run the explainer
    before = time.perf_counter()
    cfs = Explainer(x_test, target)
    after = time.perf_counter()

    lof = torch.cat([-torch.from_numpy(LOF0.score_samples(cfs.cpu()[target.view(-1) == -1].numpy())),
          -torch.from_numpy(LOF1.score_samples(cfs.cpu()[target.view(-1) == 1].numpy()))]).mean()
    if args.dataset == 'MNIST':
        success = (model(cfs).argmax(dim=1) == target.squeeze()).float()
    else:
        if args.model == 'DNN':
            success = (2 * model(cfs).cpu() - 1) * target > 0
        else:
            success = (model(cfs).cpu() * target > 0).float()
    sparsity = torch.logical_not(torch.isclose(cfs.cpu(), x_test)).float().sum(-1).mean()
    dist = torch.norm(cfs.cpu() - x_test, p=2, dim=-1).mean()
    kdeval = evaluate_DEs(kdes, cfs, target)
    gmmval = evaluate_DEs(gmms, cfs, target)

    string = f'Number of counterfactuals found: {success.sum().item()}/{len(success)}\n'
    string += f'2-norm: {dist}\n'
    string += f'0-norm: {sparsity}\n'
    string += f'LOF: {lof}\n'
    string += f'KDE: {kdeval}\n'
    string += f'GMM: {gmmval}\n'
    string += f'Time: {after - before}s\n'

    filename = f'{args.dataset}_{args.alg}_{args.stepsize}_{args.sparsity}_{args.plausibility}_{args.n_neighbors}_{args.prox}_{args.model}.txt'

    with open(os.path.join(args.resdir, filename), 'w') as f:
        f.write(string)