# Test the performance of different BERT models on disease term matching.
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import torchvision
import warnings
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

torchvision.disable_beta_transforms_warning()
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer_sapbert = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
model_sapbert = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device)

tokenizer_biobert = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
model_biobert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2").to(device)

tokenizer_clinicalbert = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model_clinicalbert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

def get_embedding(text, tokenizer, model, pooling_method="mean"):
    inputs = tokenizer(text, return_tensors="pt", truncation=False, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    if pooling_method == "mean":
        return outputs.last_hidden_state.mean(dim=1)
    elif pooling_method == "cls":
        return outputs.last_hidden_state[:,0,:].detach()
    else:
        raise ValueError(f"Invalid pooling method: {pooling_method}")

def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2).item()   

#df = pd.read_excel('disease_map/manual_TEST.xlsx') 
df = pd.read_excel('disease_map/GPT_TEST.xlsx') 
labels = df['signature']

# sapbert_mean
predictions_sapbert_mean = []
for _, row in df.iterrows():
    embedding1 = get_embedding(row['term1'], tokenizer_sapbert, model_sapbert, pooling_method="mean")
    embedding2 = get_embedding(row['term2'], tokenizer_sapbert, model_sapbert, pooling_method="mean")
    similarity = cosine_similarity(embedding1, embedding2)
    predictions_sapbert_mean.append(similarity)
auc_sapbert_mean = roc_auc_score(labels, predictions_sapbert_mean)
print("AUROC of sapbert_mean:", auc_sapbert_mean)

# sapbert_cls
predictions_sapbert_cls = []
for _, row in df.iterrows():
    embedding1 = get_embedding(row['term1'], tokenizer_sapbert, model_sapbert, pooling_method="cls")
    embedding2 = get_embedding(row['term2'], tokenizer_sapbert, model_sapbert, pooling_method="cls")
    similarity = cosine_similarity(embedding1, embedding2)
    predictions_sapbert_cls.append(similarity)
auc_sapbert_cls = roc_auc_score(labels, predictions_sapbert_cls)
print("AUROC of sapbert_cls:", auc_sapbert_cls)

# biobert_mean
predictions_biobert_mean = []
for _, row in df.iterrows():
    embedding1 = get_embedding(row['term1'], tokenizer_biobert, model_biobert, pooling_method="mean")
    embedding2 = get_embedding(row['term2'], tokenizer_biobert, model_biobert, pooling_method="mean")
    similarity = cosine_similarity(embedding1, embedding2)
    predictions_biobert_mean.append(similarity)
auc_biobert_mean = roc_auc_score(labels, predictions_biobert_mean)
print("AUROC of biobert_mean:", auc_biobert_mean)

# biobert_cls
predictions_biobert_cls = []
for _, row in df.iterrows():
    embedding1 = get_embedding(row['term1'], tokenizer_biobert, model_biobert, pooling_method="cls")
    embedding2 = get_embedding(row['term2'], tokenizer_biobert, model_biobert, pooling_method="cls")
    similarity = cosine_similarity(embedding1, embedding2)
    predictions_biobert_cls.append(similarity)
auc_biobert_cls = roc_auc_score(labels, predictions_biobert_cls)
print("AUROC of biobert_cls:", auc_biobert_cls)

# clinicalbert_mean
predictions_clinicalbert_mean = []
for _, row in df.iterrows():
    embedding1 = get_embedding(row['term1'], tokenizer_clinicalbert, model_clinicalbert, pooling_method="mean")
    embedding2 = get_embedding(row['term2'], tokenizer_clinicalbert, model_clinicalbert, pooling_method="mean")
    similarity = cosine_similarity(embedding1, embedding2)
    predictions_clinicalbert_mean.append(similarity)
auc_clinicalbert_mean = roc_auc_score(labels, predictions_clinicalbert_mean)
print("AUROC of clinicalbert_mean:", auc_clinicalbert_mean)

# clinicalbert_cls
predictions_clinicalbert_cls = []
for _, row in df.iterrows():
    embedding1 = get_embedding(row['term1'], tokenizer_clinicalbert, model_clinicalbert, pooling_method="cls")
    embedding2 = get_embedding(row['term2'], tokenizer_clinicalbert, model_clinicalbert, pooling_method="cls")
    similarity = cosine_similarity(embedding1, embedding2)
    predictions_clinicalbert_cls.append(similarity)
auc_clinicalbert_cls = roc_auc_score(labels, predictions_clinicalbert_cls)
print("AUROC of clinicalbert_cls:", auc_clinicalbert_cls)

auroc_results = {
    "sapbert_mean": auc_sapbert_mean,
    "sapbert_cls": auc_sapbert_cls,
    "biobert_mean": auc_biobert_mean,
    "biobert_cls": auc_biobert_cls,
    "clinicalbert_mean": auc_clinicalbert_mean,
    "clinicalbert_cls": auc_clinicalbert_cls
}


best_model = max(auroc_results, key=auroc_results.get)
best_predictions = eval(f"predictions_{best_model}")

fpr, tpr, thresholds = roc_curve(labels, best_predictions)
optimal_idx = (tpr - fpr).argmax()
optimal_threshold = thresholds[optimal_idx]

print(f"best_model: {best_model} with AUROC: {auroc_results[best_model]}")
print("best_threshold:", optimal_threshold)

plt.rcParams.update({
    "font.size": 16,        
    "axes.labelsize": 18,   
    "axes.titlesize": 20,   
    "xtick.labelsize": 14,  
    "ytick.labelsize": 14,  
    "legend.fontsize": 14   
})
plt.figure(figsize=(10, 8))

def plot_roc_curve(predictions, label, color):
    fpr, tpr, _ = roc_curve(labels, predictions)
    plt.plot(fpr, tpr, label=f'{label} (AUROC = {roc_auc_score(labels, predictions):.3f})', color=color)

plot_roc_curve(predictions_sapbert_mean, 'sapbert_mean', 'b')
plot_roc_curve(predictions_sapbert_cls, 'sapbert_cls', 'g')
plot_roc_curve(predictions_biobert_mean, 'biobert_mean', 'r')
plot_roc_curve(predictions_biobert_cls, 'biobert_cls', 'c')
plot_roc_curve(predictions_clinicalbert_mean, 'clinicalbert_mean', 'm')
plot_roc_curve(predictions_clinicalbert_cls, 'clinicalbert_cls', 'y')

plt.plot([0, 1], [0, 1], 'k--', label='Random')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curves for Different BERT Models on GPT_TEST\n(Threshold = {optimal_threshold:.3f})')
plt.legend(loc='lower right')
plt.tight_layout()  
plt.show()