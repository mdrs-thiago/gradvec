import torch
import torch.nn.functional as F
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from sklearn.mixture import GaussianMixture

from tqdm import tqdm 

def msp(model, loader, device):
    '''
    MSP for OOD Detection.

    Returns the MSP score for a given dataset.
    '''

    msp_ = []
    for b in tqdm(loader):

        with torch.no_grad():
            #batch.to(device)
            input_ids = b['pixel_values'].to(device)
            logits = model(input_ids)
            out = F.softmax(logits.logits, dim=-1)
            _msp, _ = torch.max(out,dim=-1)
            msp_.extend(_msp.detach().cpu().numpy())

    return np.array(msp_)

def energy(model, loader, device, temp=1):

    energy_ = []
    for b in tqdm(loader):

        with torch.no_grad():
            input_ids = b['pixel_values'].to(device)
            logits = model(input_ids)
            out = temp*torch.logsumexp(logits.logits/temp,dim=1)
            _energy = out.data.cpu().numpy()
            energy_.extend(_energy)

    return np.array(energy_)


def gradnorm(model, loader, device, temp=1):

    confs = []
    for b in tqdm(loader):

        input_ids = b['pixel_values'].to(device)
        labels = b['labels'].cpu().numpy()
        logits = model(input_ids)        
        
        targets = torch.ones((logits.logits.shape[0], logits.logits.shape[1])).cuda()
        outputs = logits.logits / temp
        
        loss1 = torch.sum(-targets * F.log_softmax(outputs, dim=-1), dim=-1)


        #loss = torch.mean(torch.sum(-targets * F.log_softmax(outputs), dim=-1))
        for loss in loss1:  
          model.zero_grad()
          loss.backward(retain_graph=True)

          layer_grad = model.classifier.weight.grad.data
          layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
          confs.append(layer_grad_norm)
              
    return np.array(confs)


class GradVec():

    def __init__(self, n_components = 3, temp=1.0, n_classes=1, computed_grad=False):
        self.components = n_components 
        self.temp = temp
        self.n_classes = n_classes
        self.computed_grad = computed_grad
        
    def get_gradients(self,model, loader, device, data_augmentation = False):

        confs = []
        labels_ = []
        if data_augmentation:
            epochs = 5
        else:
            epochs = 1
        
        for e in range(epochs):
            for b in tqdm(loader):

                input_ids = b['pixel_values'].to(device)
                labels = b['labels'].cpu().numpy()

                logits = model(input_ids)        
                
                targets = (torch.ones((logits.logits.shape[0], logits.logits.shape[1]))/self.n_classes).cuda()
                outputs = logits.logits / self.temp
                
                loss1 = torch.sum(-targets * F.log_softmax(outputs, dim=-1), dim=-1)
                
                for loss in loss1:  
                    model.zero_grad()
                    loss.backward(retain_graph=True)

                    layer_grad = model.classifier.weight.grad.data
                    layer_vec = torch.mean(torch.abs(layer_grad), dim=0).cpu().numpy()
                    confs.append(layer_vec)

                labels_.extend(labels)
                            
        
        return np.array(confs), np.array(labels_)


    def fit_PCA(self, model, loader, device, labels=None, data_augmentation = False):
        self.pca_ = {} 

        if not self.computed_grad:
            in_scores, labels = self.get_gradients(model, loader, device, data_augmentation = data_augmentation)
        else: 
            in_scores = loader 

        for i in np.unique(labels):
            pca = PCA(n_components=self.components)
            
            X_fit = in_scores[labels==i]
            pca.fit(X_fit)
            self.pca_[f'pca_{i}'] = pca

    def get_PCA_scores(self,model, loader, device):

        if not self.computed_grad:
            grads, _ = self.get_gradients(model, loader, device)
        else:
            grads = loader
        scores_ = []
        for _, estimator in self.pca_.items():
            scores = estimator.score_samples(grads)
            scores_.append(scores)
        
        grad_scores = np.max(np.array(scores_), axis=0)

        return grad_scores


    def fit_MCD(self, model, loader, device, labels=None):
        
        self.pca = PCA(n_components=self.components)

        if not self.computed_grad:
            in_grads, labels = self.get_gradients(model, loader, device)
        else:
            in_grads = loader

        X_dense = self.pca.fit_transform(in_grads)
        self.mcd_ = {}
        for i in np.unique(labels):
            
            X_fit = X_dense[labels==i]
            minCov = MinCovDet().fit(X_fit)
            self.mcd_[f'mcd_{i}'] = minCov

    def get_MCD_scores(self, model, loader, device):

        if not self.computed_grad:
            grads, _ = self.get_gradients(model, loader, device)
        else:
            grads = loader

        pca_grads = self.pca.transform(grads)

        scores_ = []
        for _, estimator in self.mcd_.items():
            scores = estimator.score_samples(pca_grads)
            scores_.append(scores)
        
        grad_scores = np.max(np.array(scores_), axis=0)
        return grad_scores

    def get_MCD_mahalanobis(self, model, loader, device):
        if not self.computed_grad:
            grads, _ = self.get_gradients(model, loader, device)
        else:
            grads = loader

        pca_grads = self.pca.transform(grads)

        scores_ = []
        for _, estimator in self.mcd_.items():
            scores = estimator.mahalanobis(pca_grads)
            scores_.append(scores)
        
        grad_scores = np.min(np.array(scores_), axis=0)
        return -grad_scores

    def fit_GMM(self, model, loader, device, labels = None, mixture=3):
        self.pca = PCA(n_components=self.components)

        if not self.computed_grad:
            in_grads, labels = self.get_gradients(model, loader, device)
        
        else:
            in_grads = loader

        X_dense = self.pca.fit_transform(in_grads)
        self.gmm = GaussianMixture(n_components=mixture).fit(X_dense)

    def get_GMM_scores(self,model, loader, device):

        if not self.computed_grad:
            grads, _ = self.get_gradients(model, loader, device)
        else:
            grads = loader
        
        X_dense = self.pca.transform(grads)
        grad_scores = self.gmm.score_samples(X_dense)

        return grad_scores




class OpenPCS():
    def __init__(self, n_components=3, computed_states=False):
        self.components = n_components
        self.computed_states = computed_states

    def get_activations(self, model, loader, device):

        activations_ = []
        labels_ = []

        for b in tqdm(loader):

            input_ids = b['pixel_values'].to(device)
            labels = b['labels'].cpu().numpy()

            output = model(input_ids)
            activation = output[1][-1]
            activation = torch.mean(activation,dim=1)
            
            activations_.extend(activation.detach().cpu().numpy())

            labels_.extend(labels)

        return np.array(activations_), np.array(labels_)

    def fit_PCA(self, model, loader, device, in_scores=None, labels=None):
        self.pca_ = {} 

        if not self.computed_states:
            in_scores, labels = self.get_activations(model, loader, device)

        for i in np.unique(labels):
            pca = PCA(n_components=self.components)
            
            X_fit = in_scores[labels==i]
            pca.fit(X_fit)
            self.pca_[f'pca_{i}'] = pca

    def get_scores(self,model, loader, device):

        if not self.computed_states:
            states, _ = self.get_activations(model, loader, device)
        else:
            states = loader
        
        scores_ = []
        for _, estimator in self.pca_.items():
            scores = estimator.score_samples(states)
            scores_.append(scores)
        
        grad_scores = np.max(np.array(scores_), axis=0)

        return grad_scores

class OpenPCSPlus(OpenPCS):
    def fit_PCA(self, model, loader, device, in_scores = None, labels=None):
        self.pca_ = {} 
        if not self.computed_states:
            in_scores, labels = self.get_activations(model, loader, device)
        
        for i in np.unique(labels):
            pca = PCA(n_components=self.components, whiten=True)
            
            X_fit = in_scores[labels==i]
            pca.fit(X_fit)
            self.pca_[f'pca_{i}'] = pca