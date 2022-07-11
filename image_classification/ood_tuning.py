from ood_metrics import get_measures
import torch 
from models import AutoBERT
from dataloader import transform_dataset
from ood_methods import *
import argparse 
import pandas as pd 
from tqdm import tqdm

from ood_metrics import get_measures
import torch 
from models import AutoBERT
from dataloader import create_loader
from ood_methods import *
import argparse 
import pandas as pd 

def run_ood(model, test_loader, ood1_test_loader, device, ood_method, temp=None, name='', save_scores=False):
  
  if temp:
    in_scores = ood_method(model, test_loader, device, temp=temp)
    out_scores = ood_method(model,ood1_test_loader, device, temp=temp)
  else:
    in_scores = ood_method(model, test_loader, device)
    out_scores = ood_method(model,ood1_test_loader, device)


  in_examples = in_scores.reshape((-1, 1))
  out_examples = out_scores.reshape((-1, 1))

  if save_scores:
    in_df = pd.DataFrame(in_examples)
    in_df.to_csv(f'in_{name}', index=None)

    out_df = pd.DataFrame(out_examples)
    out_df.to_csv(f'out_{name}', index=None)


  auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
  print(f'AUROC = {auroc} - AUPR-In = {aupr_in} - AUPR-Out = {aupr_out} - FPR95 = {fpr95}')
  return auroc, aupr_in, aupr_out, fpr95 


def experiment(args):

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  model_ = AutoBERT(model_name = args.model_name, device = device, show_hidden_states=args.hidden_states)
  model_.freeze_layers()
  
  model = model_.model
  in_train_loader, in_test_loader = create_loader(args.model_name, args.in_dataset, ID=True)
  
  out_test_loader = create_loader(args.model_name, args.out_dataset)

  model.eval();

  if 'gradvecpca' in args.ood:
    gradvec = GradVec(n_components=args.n_components, temp=args.temp, n_classes=args.n_classes, computed_grad=True)
    
    print('Getting gradients')
    in_train_grads, in_train_labels = gradvec.get_gradients(model, in_train_loader, device)
    in_test_grads, _ = gradvec.get_gradients(model, in_test_loader, device)
    out_test_grads, _ = gradvec.get_gradients(model, out_test_loader, device)

    res = pd.DataFrame(in_train_grads)
    res.to_csv(f'in_train_grads_{args.in_dataset}_{args.out_dataset}.csv', index=None)
    res = pd.DataFrame(in_test_grads)
    res.to_csv(f'in_test_grads_{args.in_dataset}_{args.out_dataset}.csv', index=None)
    res = pd.DataFrame(out_test_grads)
    res.to_csv(f'out_test_grads_{args.in_dataset}_{args.out_dataset}.csv', index=None)



    print('Computing GradPCA')
    results = []
    for n in tqdm(range(3,4)):
      gradvec = GradVec(n_components=n, temp=args.temp, n_classes=args.n_classes, computed_grad=True)
      gradvec.fit_PCA(model, in_train_grads, device, labels=in_train_labels)
      gradvec_res = run_ood(model, in_test_grads, out_test_grads, device, gradvec.get_PCA_scores)
      
      res = [f'gradvec_{n}', gradvec_res[0], gradvec_res[1], gradvec_res[2], gradvec_res[3]]
      results.append(res)
    
    res_pd = pd.DataFrame(results)
    res_pd.to_csv(f'gradpca_results_{args.in_dataset}_{args.out_dataset}.csv')

  if 'gradvecmaha' in args.ood:
    print('Computing GradMaha')
    results = []
    for n in tqdm(range(2,50)):
      gradvec = GradVec(n_components=n, temp=args.temp, n_classes=args.n_classes, computed_grad=True)
      gradvec.fit_MCD(model, in_train_grads, device, labels=in_train_labels)
      gradvec_res = run_ood(model, in_test_grads, out_test_grads, device, gradvec.get_MCD_mahalanobis)
      res = [f'gradvec_{n}', gradvec_res[0], gradvec_res[1], gradvec_res[2], gradvec_res[3]]
      results.append(res)
    
    res_pd = pd.DataFrame(results)
    res_pd.to_csv('gradmaha_results.csv')

  if 'gradvecgmm' in args.ood:
    print('Computing GradGMM')
    results = []
    for n in tqdm(range(3,4)):
      for m in range(3,4):
        gradvec = GradVec(n_components=n, temp=args.temp, n_classes=args.n_classes, computed_grad=True)
        gradvec.fit_GMM(model, in_train_grads, device, labels=in_train_labels, mixture=m)
        gradvec_res = run_ood(model, in_test_grads, out_test_grads, device, gradvec.get_GMM_scores)
        res = [f'gradvec_{n}_{m}', gradvec_res[0], gradvec_res[1], gradvec_res[2], gradvec_res[3]]
        results.append(res)
      
    res_pd = pd.DataFrame(results)
    res_pd.to_csv(f'gradgmm_results_{args.in_dataset}_{args.out_dataset}.csv')

  if 'openpcs' in args.ood:
    openpcs = OpenPCS(n_components=args.n_components, computed_states=True)

    print('Getting activations')    
    in_train_act, in_train_labels = openpcs.get_activations(model, in_train_loader, device)
    in_test_act, _ = openpcs.get_activations(model, in_test_loader, device)
    out_test_act, _ = openpcs.get_activations(model, out_test_loader, device)

    res = pd.DataFrame(in_train_act)
    res.to_csv(f'in_train_act_{args.in_dataset}_{args.out_dataset}.csv', index=None)
    res = pd.DataFrame(in_test_act)
    res.to_csv(f'in_test_act_{args.in_dataset}_{args.out_dataset}.csv', index=None)
    res = pd.DataFrame(out_test_act)
    res.to_csv(f'out_test_act_{args.in_dataset}_{args.out_dataset}.csv', index=None)


    results = []
    
    for n in tqdm(range(3,4)):
      openpcs = OpenPCS(n_components=n, computed_states=True)
      openpcs.fit_PCA(model, in_train_loader, device, in_scores=in_train_act, labels= in_train_labels)
      openpcs_res = run_ood(model, in_test_act, out_test_act, device, openpcs.get_scores)
      res = [f'openpcs_{n}',openpcs_res[0], openpcs_res[1], openpcs_res[2], openpcs_res[3]]
      results.append(res)

    res_pd = pd.DataFrame(results)
    res_pd.to_csv(f'openpcs_results_{args.in_dataset}_{args.out_dataset}.csv')

    results = []
    print('OpenPCS++ results')
    for n in tqdm(range(3,4)):
      openpcs = OpenPCSPlus(n_components=n, computed_states=True)
      openpcs.fit_PCA(model, in_train_loader, device, in_scores=in_train_act, labels= in_train_labels)
      openpcs_res = run_ood(model, in_test_act, out_test_act, device, openpcs.get_scores)
      res = [f'openpcs_{n}',openpcs_res[0], openpcs_res[1], openpcs_res[2], openpcs_res[3]]
      results.append(res)

    res_pd = pd.DataFrame(results)
    res_pd.to_csv(f'openpcsplus_results_{args.in_dataset}_{args.out_dataset}.csv')

  
  #res_OOD = pd.DataFrame([msp_res, energy_res, gradvec_res, gradvec_gmm, openpcs_res, openpcsplus_res], columns = ['AUROC','AUPR-In','AUPR-Out','FPR95'])
  #res_OOD.to_csv(f'results_{args.in_dataset}_{args.out_dataset}.csv', index=False)

  return gradvec_res 
  
if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='List of parameters for OOD detection in CV tasks')
  parser.add_argument('--ood', choices=['msp','energy','gradnorm','gradvecpca','gradvecgmm','openpcs'], nargs='+', type=str, default=['msp','energy','gradnorm','gradvecpca', 'gradvecgmm','openpcs'])
  parser.add_argument('--in_dataset',type=str,default='cifar10')
  parser.add_argument('--out_dataset', type=str,default='svhn')
  parser.add_argument('--model_name', type=str, default='tanlq/vit-base-patch16-224-in21k-finetuned-cifar10')
  parser.add_argument('--hidden_states',type=bool,default=False)
  parser.add_argument('--n_components', type=int, default=10)
  parser.add_argument('--temp', type=float, default=1.0)
  parser.add_argument('--n_classes', type=int, default=10)
  parser.add_argument('--save_score',type=bool,default=True)
  

  experiment(parser.parse_args())

def experiment(args):

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  model_ = AutoBERT(model_name = args.model_name, device = device, show_hidden_states=args.hidden_states)
  model_.freeze_layers()
  
  model = model_.model
  in_train_loader, in_test_loader = transform_dataset(args.in_dataset, model_.tokenizer)
  _, out_test_loader = transform_dataset(args.out_dataset, model_.tokenizer)

  model.eval();

  if 'gradvecpca' in args.ood:
    gradvec = GradVec(n_components=args.n_components, temp=args.temp, n_classes=args.n_classes, computed_grad=True)
    
    print('Getting gradients')
    in_train_grads, in_train_labels = gradvec.get_gradients(model, in_train_loader, device)
    in_test_grads, _ = gradvec.get_gradients(model, in_test_loader, device)
    out_test_grads, _ = gradvec.get_gradients(model, out_test_loader, device)

    print('Computing GradPCA')
    results = []
    for n in tqdm(range(2,70)):
      gradvec = GradVec(n_components=n, temp=args.temp, n_classes=args.n_classes, computed_grad=True)
      gradvec.fit_PCA(model, in_train_grads, device, labels=in_train_labels)
      gradvec_res = run_ood(model, in_test_grads, out_test_grads, device, gradvec.get_PCA_scores)
      res = [f'gradvec_{n}', gradvec_res[0], gradvec_res[1], gradvec_res[2], gradvec_res[3]]
      results.append(res)
    
    res_pd = pd.DataFrame(results)
    res_pd.to_csv(f'gradpca_results_{args.in_dataset}_{args.out_dataset}.csv')

  if 'gradvecmaha' in args.ood:
    print('Computing GradMaha')
    results = []
    for n in tqdm(range(2,50)):
      gradvec = GradVec(n_components=n, temp=args.temp, n_classes=args.n_classes, computed_grad=True)
      gradvec.fit_MCD(model, in_train_grads, device, labels=in_train_labels)
      gradvec_res = run_ood(model, in_test_grads, out_test_grads, device, gradvec.get_MCD_mahalanobis)
      res = [f'gradvec_{n}', gradvec_res[0], gradvec_res[1], gradvec_res[2], gradvec_res[3]]
      results.append(res)
    
    res_pd = pd.DataFrame(results)
    res_pd.to_csv('gradmaha_results.csv')

  if 'gradvecgmm' in args.ood:
    print('Computing GradGMM')
    results = []
    for n in tqdm(range(2,70)):
      for m in range(2,15):
        gradvec = GradVec(n_components=n, temp=args.temp, n_classes=args.n_classes, computed_grad=True)
        gradvec.fit_GMM(model, in_train_grads, device, labels=in_train_labels, mixture=m)
        gradvec_res = run_ood(model, in_test_grads, out_test_grads, device, gradvec.get_GMM_scores)
        res = [f'gradvec_{n}_{m}', gradvec_res[0], gradvec_res[1], gradvec_res[2], gradvec_res[3]]
        results.append(res)
      
    res_pd = pd.DataFrame(results)
    res_pd.to_csv(f'gradgmm_results_{args.in_dataset}_{args.out_dataset}.csv')

  if 'openpcs' in args.ood:
    openpcs = OpenPCS(n_components=args.n_components, computed_states=True)

    print('Getting activations')    
    in_train_act, in_train_labels = openpcs.get_activations(model, in_train_loader, device)
    in_test_act, _ = openpcs.get_activations(model, in_test_loader, device)
    out_test_act, _ = openpcs.get_activations(model, out_test_loader, device)

    results = []
    
    for n in tqdm(range(2,70)):
      openpcs = OpenPCS(n_components=n, computed_states=True)
      openpcs.fit_PCA(model, in_train_loader, device, in_scores=in_train_act, labels= in_train_labels)
      openpcs_res = run_ood(model, in_test_act, out_test_act, device, openpcs.get_scores)
      res = [f'openpcs_{n}',openpcs_res[0], openpcs_res[1], openpcs_res[2], openpcs_res[3]]
      results.append(res)

    res_pd = pd.DataFrame(results)
    res_pd.to_csv(f'openpcs_results_{args.in_dataset}_{args.out_dataset}.csv')

    results = []
    print('OpenPCS++ results')
    for n in tqdm(range(2,70)):
      openpcs = OpenPCSPlus(n_components=n, computed_states=True)
      openpcs.fit_PCA(model, in_train_loader, device, in_scores=in_train_act, labels= in_train_labels)
      openpcs_res = run_ood(model, in_test_act, out_test_act, device, openpcs.get_scores)
      res = [f'openpcs_{n}',openpcs_res[0], openpcs_res[1], openpcs_res[2], openpcs_res[3]]
      results.append(res)

    res_pd = pd.DataFrame(results)
    res_pd.to_csv(f'openpcsplus_results_{args.in_dataset}_{args.out_dataset}.csv')

    
if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='List of parameters for OOD detection in NLP tasks')
  parser.add_argument('--ood', choices=['msp','energy','gradnorm','gradvecpca','gradvecmcd','gradvecgmm','openpcs'], nargs='+', type=str, default=['msp','energy','gradnorm','gradvec','openpcs'])
  parser.add_argument('--in_dataset',type=str,default='emotion')
  parser.add_argument('--out_dataset', type=str,default='ag_news')
  parser.add_argument('--model_name', type=str, default='bhadresh-savani/distilbert-base-uncased-emotion')
  parser.add_argument('--hidden_states',type=bool,default=False)
  parser.add_argument('--n_components', type=int, default=3)
  parser.add_argument('--temp', type=float, default=1.0)
  parser.add_argument('--n_classes', type=int, default=1)

  experiment(parser.parse_args())