from ood_metrics import get_measures
import torch 
from models import AutoBERT
from dataloader import transform_dataset
from ood_methods import *
import argparse 
import pandas as pd 

def run_ood(model, test_loader, device, ood_method, temp=None):
  
  if temp:
    in_scores = ood_method(model, test_loader, device, temp=temp)
  else:
    in_scores = ood_method(model, test_loader, device)
  

  in_examples = in_scores.reshape((-1, 1))
  return in_examples

def experiment(args):

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  model_ = AutoBERT(model_name = args.model_name, device = device, show_hidden_states=args.hidden_states)
  model_.freeze_layers()
  
  model = model_.model
  in_train_loader, in_test_loader = transform_dataset(args.in_dataset, model_.extractor)
  
  model.eval();

  #MSP test
  if 'msp' in args.ood:
    print('MSP Test')
    msp_res = run_ood(model, in_test_loader, device, msp)
    res_pd = pd.DataFrame(msp_res)
    res_pd.to_csv(f'msp_{args.in_dataset}_{args.out_dataset}.csv')

    
  #Energy
  if 'energy' in args.ood:
    print('Energy test')
    energy_res = run_ood(model, in_test_loader, device, energy, temp=args.temp)
    res_pd = pd.DataFrame(energy_res)
    res_pd.to_csv(f'energy_{args.in_dataset}_{args.out_dataset}.csv')

  #GradVec 
  if 'gradvecpca' in args.ood:
    print('GradVec test')

    gradvec = GradVec(n_components=args.n_components, temp=args.temp, n_classes=args.n_classes)
    gradients, labels = gradvec.get_gradients(model, in_train_loader, device)
    res_pd = pd.DataFrame(gradients)
    res_pd.to_csv(f'gradvec_train_{args.in_dataset}_{args.out_dataset}.csv')
    res_pd = pd.DataFrame(labels)
    res_pd.to_csv(f'gradvec_train_labels_{args.in_dataset}_{args.out_dataset}.csv')

    gradients_, labels_ = gradvec.get_gradients(model, in_test_loader, device)
    res_pd = pd.DataFrame(gradients_)
    res_pd.to_csv(f'gradvec_test_{args.in_dataset}_{args.out_dataset}.csv')
    res_pd = pd.DataFrame(labels_)
    res_pd.to_csv(f'gradvec_test_labels_{args.in_dataset}_{args.out_dataset}.csv')


    #gradvec.fit_PCA(model, in_train_loader, device)
    #gradvec_res = run_ood(model, in_train_loader, device, gradvec.get_PCA_scores)
    #gradvec_res = run_ood(model, in_test_loader, device, gradvec.get_PCA_scores)


  #OpenPCS
  if 'openpcs' in args.ood:
    print('OpenPCS test') 
    openpcs = OpenPCS(n_components=args.n_components)
    openpcs.fit_PCA(model, in_train_loader, device)
    openpcs_res = run_ood(model, in_test_loader, out_test_loader, device, openpcs.get_scores)

  return msp_res, energy_res, gradnorm_res, gradients 
  
if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='List of parameters for OOD detection in CV tasks')
  parser.add_argument('--ood', choices=['msp','energy','gradnorm','gradvecpca','gradvecmcd','gradvecgmm','openpcs'], nargs='+', type=str, default=['msp','energy','gradnorm','gradvecpca'])
  parser.add_argument('--in_dataset',type=str,default='cifar10')
  parser.add_argument('--out_dataset', type=str,default='svhn')
  parser.add_argument('--model_name', type=str, default='tanlq/vit-base-patch16-224-in21k-finetuned-cifar10')
  parser.add_argument('--hidden_states',type=bool,default=False)
  parser.add_argument('--n_components', type=int, default=3)
  parser.add_argument('--temp', type=float, default=1.0)
  parser.add_argument('--n_classes', type=int, default=10)

  experiment(parser.parse_args())