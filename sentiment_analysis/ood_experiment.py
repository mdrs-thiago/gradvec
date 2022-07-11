from ood_metrics import get_measures
import torch 
from models import AutoBERT
from dataloader import transform_dataset
from ood_methods import *

def run_ood(model, test_loader, ood1_test_loader, device, ood_method):
  in_scores = ood_method(model, test_loader, device)
  out_scores = ood_method(model,ood1_test_loader, device)

  in_examples = in_scores.reshape((-1, 1))
  out_examples = out_scores.reshape((-1, 1))

  auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
  print(f'AUROC = {auroc} - AUPR-In = {aupr_in} - AUPR-Out = {aupr_out} - FPR95 = {fpr95}')
  return auroc, aupr_in, aupr_out, fpr95 


def experiment(in_dataset='emotion', out_dataset='ag_news', model_name = "bhadresh-savani/distilbert-base-uncased-emotion"):

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  model_ = AutoBERT(model_name = model_name, device = device)
  model_.freeze_layers()
  
  model = model_.model
  in_train_loader, in_test_loader = transform_dataset(in_dataset, model_.tokenizer)
  out_train_loader, out_test_loader = transform_dataset(out_dataset, model_.tokenizer)

  model.eval();

  #MSP test
  print('MSP Test')
  msp_res = run_ood(model, in_test_loader, out_test_loader, device, msp)
  
  #Energy
  print('Energy test')
  energy_res = run_ood(model, in_test_loader, out_test_loader, device, energy)
  
  #Grad Norm
  print('GradNorm test')
  gradnorm_res = run_ood(model, in_test_loader, out_test_loader, device, gradnorm)

  #GradVec 
  print('GradVec test')

  gradvec = GradVec()
  gradvec.fit_PCA(model, in_train_loader, device, temp=1)

  gradvec_res = run_ood(model, in_test_loader, out_test_loader, device, gradvec.get_scores)

  return msp_res, energy_res, gradnorm_res, gradvec_res 
  
if __name__ == "__main__":
  experiment()