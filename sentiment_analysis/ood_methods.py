import torch
import torch.nn.functional as F
import numpy as np 
from sklearn.decomposition import PCA

def msp(model, loader, device):
    '''
    MSP for OOD Detection.

    Returns the MSP score for a given dataset.
    '''

    msp_ = []
    for b, batch in enumerate(loader):
        with torch.no_grad():
            #batch.to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            out = F.softmax(logits.logits, dim=-1)
            _msp, _ = torch.max(out,dim=-1)
            msp_.extend(_msp.detach().cpu().numpy())
        if b%100 == 0:
          print(f'Processing batch {b}')
    return np.array(msp_)

'''
def gradnorm(model, loader, n_classes=1,device='cpu', temp=1):

    gradnorm_ = []
    for batch in loader:

        model.zero_grad()

        logits = model(**batch)
        
        targets = torch.ones((batch.shape[0], n_classes)).to(device)
        outputs = outputs/temp
        
        loss = torch.mean(torch.sum(-targets * F.Softmax(outputs), dim=-1))
        loss.backward()

        layer_grad = model['layer'].weight.grad.data
        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        gradnorm_.extend(layer_grad_norm)

    return np.array(gradnorm_)'''

def energy(model, loader, device, temp=1):

    energy_ = []
    for batch in loader:

        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            out = temp*torch.logsumexp(logits.logits/temp,dim=1)
            _energy = out.data.cpu().numpy()
            energy_.extend(_energy)

    return np.array(energy_)


def gradnorm(model, loader, device, temp=1):

    confs = []
    for i,batch in enumerate(loader):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)        
        
        targets = torch.ones((logits.logits.shape[0], logits.logits.shape[1])).cuda()
        outputs = logits.logits / temp
        
        loss1 = torch.sum(-targets * F.log_softmax(outputs), dim=-1)


        #loss = torch.mean(torch.sum(-targets * F.log_softmax(outputs), dim=-1))
        for loss in loss1:  
          model.zero_grad()
          loss.backward(retain_graph=True)

          layer_grad = model.classifier.weight.grad.data
          layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
          confs.append(layer_grad_norm)
        
        if i % 50 == 0:
          print(f'Running batch {i}')
      
    return np.array(confs)

class OpenPCS():
    def __init__(self, n_components=3):
        self.components = n_components

    def get_activations(self, model, loader, device):

        activations_ = []
        labels_ = []

        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            activations_.append(output[1][-1].squeeze(0).flatten())

            labels_.extend(labels)

        return np.array(activations_), np.array(labels_)


class GradVec():

    def __init__(self, n_components = 3):
        self.components = n_components 
        
    def get_gradients(self,model, loader, device, temp=1):

        confs = []
        labels_ = []
        for i,batch in enumerate(loader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)        
            
            targets = torch.ones((logits.logits.shape[0], logits.logits.shape[1])).cuda()
            outputs = logits.logits / temp
            
            loss1 = torch.sum(-targets * F.log_softmax(outputs), dim=-1)
            
            for loss in loss1:  
                model.zero_grad()
                loss.backward(retain_graph=True)

                layer_grad = model.classifier.weight.grad.data
                layer_vec = torch.mean(torch.abs(layer_grad), dim=0).cpu().numpy()
                confs.append(layer_vec)

            labels_.extend(labels)
            
            if i % 5 == 0:
                print(f'Running batch {i}')
            
        
        return np.array(confs), np.array(labels_)


    def fit_PCA(self, model, loader, device, temp=1):
        self.pca_ = {} 
        in_scores, labels = self.get_gradients(model, loader, device, temp=temp)

        for i in np.unique(labels):
            pca = PCA(n_components=3)
            
            X_fit = in_scores[labels==i]
            pca.fit(X_fit)
            self.pca_[f'pca_{i}'] = pca

    def get_scores(self,model, loader, device, temp=1):

        grads, _ = self.get_gradients(model, loader, device, temp=temp)
        scores_ = []
        for _, estimator in self.pca_.items():
            scores = estimator.score_samples(grads)
            scores_.append(scores)
        
        grad_scores = np.max(np.array(scores_), axis=0)

        return grad_scores



class OpenPCS():
    def __init__(self, n_components=3):
        self.components = n_components

    def get_activations(self, model, loader, device):

        activations_ = []
        labels_ = []

        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            activations_.append(output[1][-1].squeeze(0).flatten())

            labels_.extend(labels)

        return np.array(activations_), np.array(labels_)

    def fit_PCA(self, model, loader, device, temp=1):
        self.pca_ = {} 
        in_scores, labels = self.get_activations(model, loader, device, temp=temp)

        for i in np.unique(labels):
            pca = PCA(n_components=3)
            
            X_fit = in_scores[labels==i]
            pca.fit(X_fit)
            self.pca_[f'pca_{i}'] = pca

    def get_scores(self,model, loader, device, temp=1):

        grads, _ = self.get_activations(model, loader, device, temp=temp)
        scores_ = []
        for _, estimator in self.pca_.items():
            scores = estimator.score_samples(grads)
            scores_.append(scores)
        
        grad_scores = np.max(np.array(scores_), axis=0)

        return grad_scores