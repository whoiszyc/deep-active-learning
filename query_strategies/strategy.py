import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.n_label = len(np.unique(Y))
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def query(self, n, logger):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train_binary(self, epoch, loader_tr, optimizer):
        self.clf.train()
        loss_func = torch.nn.BCELoss()  # ZYC
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            # loss = F.cross_entropy(out, y) # out.type()=float and y.type()=long
            loss = loss_func(out, y)  # ZYC
            loss.backward()
            optimizer.step()
        # print("Epoch: {}; Loss: {}".format(epoch, loss.item()))

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y) # out.type()=float and y.type()=long
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print("Epoch: {}; Loss: {}".format(epoch, loss.item()))

    def train(self, logger):
        n_epoch = self.args['n_epoch']
        self.clf = self.net().to(self.device)
        # optimizer = optim.SGD(self.clf.parameters(), **self.args['optimizer_args'])
        optimizer = optim.Adam(self.clf.parameters(), lr=1e-3)  # ZYC: learning rate is one of the key
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args['transform']),
                            shuffle=True, **self.args['loader_tr_args'])
        print('Now train with {} samples'.format(len(loader_tr.dataset.Y)))
        logger.info('Now train with {} samples'.format(len(loader_tr.dataset.Y)))

        for epoch in range(1, n_epoch+1):
            self._train(epoch, loader_tr, optimizer)


    def predict(self, X, Y, logger, flag=0):
        # for base prediction
        if flag == 1:
            print("Predict using untrained model")
            logger.info("Predict using untrained model")
            self.clf = self.net().to(self.device)
        else:
            print("Predict using trained model")

        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                # get maximum value along axis 1, return the values and indices
                pred = out.max(1)[1]
                P[idxs] = pred.cpu()
        return P


    def predict_prob(self, X, Y, logger):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        # here we change second tensor dim from len(np.unique(Y)) to self.n_label to fix bug (due to data imbalance)
        # But we are not sure if it is the requirement of the algorithm to define dim based on current feasture
        # Same for the following code
        probs = torch.zeros([len(Y), self.n_label])
        # logger.debug("shape of probs is {}".format(probs.shape))
        n = 0
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                # perform softmax operation along dimension 1 (label dimension)
                prob = F.softmax(out, dim=1)
                # # debug code
                # n = n + 1
                # logger.debug("at iter {} shape of prob is {}".format(n, prob.shape))
                # logger.debug("at iter {} shape of idxs is {}".format(n, idxs.shape))
                probs[idxs] = prob.cpu()
        return probs


    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), self.n_label])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        
        return probs


    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), self.n_label])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        
        return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()
        
        return embedding

