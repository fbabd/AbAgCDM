import torch
import argparse
import json 
import os

from AbAgCDM import create_single_dataloader, test , get_AbAgCDM 


if __name__=="__main__":
    # LOAD TRAINED MODEL 
    trained_model = get_AbAgCDM(model_directory="AbAgCDM") 
    model = trained_model["model"] 
    device = trained_model["device"]
    model_weights_path = trained_model["weight_path"]
    
    # ========================================================================
    # LOAD DATA & EVALUATE 
    # ========================================================================
    dataloader = create_single_dataloader(data_filepath_pq="./_data/ab_splits/il6_Q144A_disjoint.parquet") 
    
    # for batch in dataloader:
    #     print(batch.keys())
    #     break 
    
    ts_results = test(model, dataloader, device, model_weights_path) 
    
    print(ts_results)



    
# OUTPUT: 

# ================================================================================
# Test Results: (test)
# ================================================================================
# Loss: 0.2723
# Accuracy: 0.9925
# Precision: 0.8942
# Recall: 0.8692
# F1 Score: 0.8815
# AUROC: 0.9921
# AUPRC: 0.9464
# ================================================================================

# {'loss': 0.272274433295444, 'accuracy': 0.9924834636199639, 
# 'balanced_accuracy': 0.9328708340954555, 'precision': 0.8942307692307693, 
# 'recall': 0.8691588785046729, 'specificity': 0.9965827896862379, 
# 'f1': 0.8815165876777251, 'auroc': 0.9920913501319559, 'auprc': 0.9463688481345812, 
# 'tp': 93, 'fp': 11, 'tn': 3208, 'fn': 14}


# ================================================================================
# Test Results:  il6_Q144A_disjoint 
# ================================================================================
# Loss: 8.5982
# Accuracy: 0.9247
# Precision: 1.0000
# Recall: 0.8882
# F1 Score: 0.9408
# AUROC: 0.9630
# AUPRC: 0.9849
# ================================================================================

# {'loss': 8.59815959185362, 'accuracy': 0.9246861924686193, 
# 'balanced_accuracy': 0.9440993788819876, 'precision': 1.0, 
# 'recall': 0.8881987577639752, 'specificity': 1.0, 
# 'f1': 0.9407894736842105, 'auroc': 0.9629718107978977, 
# 'auprc': 0.9849276537512921, 'tp': 143, 'fp': 0, 'tn': 78, 'fn': 18}
