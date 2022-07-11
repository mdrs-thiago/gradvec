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


  auroc, aupr_in, aupr_out, fpr95, fpr, tpr = get_measures(in_examples, out_examples)
  print(f'AUROC = {auroc} - AUPR-In = {aupr_in} - AUPR-Out = {aupr_out} - FPR95 = {fpr95}')
  return auroc, aupr_in, aupr_out, fpr95, fpr, tpr


def experiment(args):

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  model_ = AutoBERT(model_name = args.model_name, device = device, show_hidden_states=args.hidden_states)
  model_.freeze_layers()
  
  model = model_.model
  in_train_loader, in_test_loader = create_loader(args.model_name, args.in_dataset, ID=True, data_augmentation = args.data_augmentation)
  
  out_test_loader = create_loader(args.model_name, args.out_dataset)

  model.eval();

  #MSP test
  if 'msp' in args.ood:
    print('MSP Test')
    msp_res = run_ood(model, in_test_loader, out_test_loader, device, msp, save_scores = args.save_score, name=f'msp_{args.in_dataset}_{args.out_dataset}.csv')
    
  #Energy
  if 'energy' in args.ood:
    print('Energy test')
    energy_res = run_ood(model, in_test_loader, out_test_loader, device, energy, temp=args.temp, save_scores = args.save_score, name=f'energy_{args.in_dataset}_{args.out_dataset}.csv')
    
  #Grad Norm
  if 'gradnorm' in args.ood:
    print('GradNorm test')
    gradnorm_res = run_ood(model, in_test_loader, out_test_loader, device, gradnorm, temp=args.temp, save_scores = args.save_score, name=f'gradnorm_{args.in_dataset}_{args.out_dataset}.csv')

  #GradVec 
  if 'gradvecpca' in args.ood:
    print('GradVec test')

    gradvec = GradVec(n_components=args.n_components, temp=args.temp, n_classes=args.n_classes)
    gradvec.fit_PCA(model, in_train_loader, device, data_augmentation = args.data_augmentation)
    gradvec_res = run_ood(model, in_test_loader, out_test_loader, device, gradvec.get_PCA_scores, save_scores = args.save_score, name=f'gradpca_{args.in_dataset}_{args.out_dataset}.csv')

  if 'gradvecmcd' in args.ood:
    print('GradVec - MCD test')

    gradvec = GradVec(n_components=args.n_components, temp=args.temp, n_classes=args.n_classes)
    gradvec.fit_MCD(model, in_train_loader, device)
    #gradvec_res = run_ood(model, in_test_loader, out_test_loader, device, gradvec.get_MCD_scores)
    print('Using mahalanobis distance')
    gradvec_maha = run_ood(model, in_test_loader, out_test_loader, device, gradvec.get_MCD_mahalanobis)
    
  if 'gradvecgmm' in args.ood:
    print('GradVec - GMM test')

    gradvec = GradVec(n_components=args.n_components, temp=args.temp, n_classes=args.n_classes)
    gradvec.fit_GMM(model, in_train_loader, device, mixture=args.n_classes)
    print('Using MLE')
    gradvec_gmm = run_ood(model, in_test_loader, out_test_loader, device, gradvec.get_GMM_scores, save_scores = args.save_score, name=f'gradgmm_{args.in_dataset}_{args.out_dataset}.csv')


  #OpenPCS
  if 'openpcs' in args.ood:

    model_ = AutoBERT(model_name = args.model_name, device = device, show_hidden_states=True)
    model_.freeze_layers()
    
    model = model_.model
    model.eval();

    print('OpenPCS test') 
    openpcs = OpenPCS(n_components=args.n_components)
    openpcs.fit_PCA(model, in_train_loader, device)
    openpcs_res = run_ood(model, in_test_loader, out_test_loader, device, openpcs.get_scores, save_scores = args.save_score, name=f'openpcs_{args.in_dataset}_{args.out_dataset}.csv')

    openpcs = OpenPCSPlus(n_components=args.n_components)
    openpcs.fit_PCA(model, in_train_loader, device)
    openpcsplus_res = run_ood(model, in_test_loader, out_test_loader, device, openpcs.get_scores, save_scores = args.save_score, name=f'openpcsp_{args.in_dataset}_{args.out_dataset}.csv')

  res_OOD = pd.DataFrame([msp_res, energy_res, gradvec_res, gradvec_gmm, openpcs_res, openpcsplus_res], columns = ['AUROC','AUPR-In','AUPR-Out','FPR95'])
  res_OOD.to_csv(f'results_{args.in_dataset}_{args.out_dataset}.csv', index=False)

  return msp_res, energy_res, gradnorm_res, gradvec_res, gradvec_gmm, openpcs_res, openpcsplus_res
  
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
  parser.add_argument('--data_augmentation', type=bool, default=False)
  

  experiment(parser.parse_args())