import pandas as pd
# import seaborn as sns
# from utils import ECEMetric, EER
from sklearn.metrics import f1_score, accuracy_score
import ast
import numpy as np
# sns.set_theme(style="darkgrid")
from matplotlib.pyplot import ylabel
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from netcal.metrics import ECE

def EER(y, y_softmax_scores, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]) :
    y = label_binarize(y, classes=classes)
    n_classes = 14
    y_softmax_scores = np.stack(y_softmax_scores, axis=0)
    total_eer = 0
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y[:, i], y_softmax_scores[:, i])
            fnr = 1 - tpr
            eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            total_eer += eer
        except:
            pass
    
    average_eer = total_eer/n_classes
    return average_eer

def Cavg(y, y_pred):
    # https://www.nist.gov/system/files/documents/2017/09/29/lre17_eval_plan-2017-09-29_v1.pdf
    # section 3.1
    ntar = 14
    cavg_1 = 0.0
    beta_1 = 1.0

    cavg_2 = 0.0
    beta_2 = 9.0

    P_FA = np.zeros((ntar,ntar)).astype(float)
    P_Miss = np.zeros((ntar,)).astype(float)

    for i, label in enumerate(y):
        pred_label = y_pred[i]
        if(label != pred_label):
            P_FA[label][pred_label] += 1
            P_Miss[label] += 1

    tar_count = np.array([y.count(i) for i in range(ntar)])
    print(P_FA)
    print(tar_count)

    # get probabilities
    P_FA /= tar_count.reshape(1, -1).transpose()
    P_Miss /= tar_count

    print(P_FA)
    print(P_Miss)

    for i in range(ntar):
        cavg_1 += P_Miss[i]
        cavg_2 += P_Miss[i]
        
        for j in range(ntar):
            cavg_1 += (beta_1/(ntar-1))*P_FA[i][j]
            cavg_2 += (beta_2/(ntar-1))*P_FA[i][j]

    cavg_1 /= ntar
    cavg_2 /= ntar
    
    c_primary = (cavg_1 + cavg_2)/2
    return c_primary
    
def ECEMetric(y, y_softmax_scores):
    y_softmax_scores = np.stack(y_softmax_scores, axis=0)
    ece = ECE(10)
    return ece.measure(y_softmax_scores, y)


set_1 = pd.read_csv('/root/Langid/LangID-LRE17/results/csv/xlsr-latent-random-1.csv')
set_2 = pd.read_csv('/root/Langid/LangID-LRE17/results/csv/xlsr-latent-random-1.csv')
set_3 = pd.read_csv('/root/Langid/LangID-LRE17/results/csv/xlsr-latent-random-1.csv')

label2num = {'ara-acm': 0, 'ara-apc': 1, 'ara-ary': 2, 'ara-arz': 3, 'eng-gbr': 4, 'eng-usg': 5, 'qsl-pol': 6, 'qsl-rus': 7, 'por-brz': 8, 'spa-car': 9, 'spa-eur': 10, 'spa-lac': 11, 'zho-cmn': 12, 'zho-nan': 13}
num2cluster = {0:1, 1:1, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3, 8:4, 9:4, 10:4, 11:4, 12:5, 13:5}

clusters = [['arabic',[0,1,2,3]], ['english',[4,5]], ['slavic',[6,7]], ['iberian',[8,9,10,11]], ['chinese',[12,13]]]


def cluster_accuracy(df):
    true_cluster = []
    predicted_cluster = []
    for i in range(len(df)):
        true_cluster.append(num2cluster[label2num[df.loc[i, 'class']]])
        predicted_cluster.append(num2cluster[label2num[df.loc[i, 'prediction']]])

    return accuracy_score(true_cluster, predicted_cluster)


def get_cluster_wise(df):
    scores = []
    for i in range(len(clusters)):
        true = []
        predicted = []
        for k in range(len(df)):
            if label2num[df.loc[k, 'class']] in clusters[i][1]:
                true.append(df.loc[k,'class'])
                predicted.append(df.loc[k, 'prediction'])
        scores.append(accuracy_score(true, predicted))
    return scores


def get_ece(df):
    probs = []
    y_true = []
    for i in range(len(df)):
        probs.append(np.array(ast.literal_eval(df.loc[i, 'probability'])))
        y_true.append(label2num[df.loc[i, 'class']])
    y_true = np.array(y_true)
    return ECEMetric(y_true, probs)


def metrics(df):
    language_true_3 = []
    language_pred_3 = []
    language_softmax_3 = []

    language_true_10 = []
    language_pred_10 = []
    language_softmax_10 = []

    language_true_30 = []
    language_pred_30 = []
    language_softmax_30 = []

    for i in range(len(df)):
        if(int(df.loc[i,'duration']) == 3):
            language_pred_3.append(df.loc[i, 'prediction'])
            language_true_3.append(df.loc[i, 'class'])
            language_softmax_3.append(ast.literal_eval(df.loc[i, 'probability']))
        if(int(df.loc[i,'duration']) == 10):
            language_pred_10.append(df.loc[i, 'prediction'])
            language_true_10.append(df.loc[i, 'class'])
            language_softmax_10.append(ast.literal_eval(df.loc[i, 'probability']))
        if(int(df.loc[i,'duration']) == 30):
            language_pred_30.append(df.loc[i, 'prediction'])
            language_true_30.append(df.loc[i, 'class'])
            language_softmax_30.append(ast.literal_eval(df.loc[i, 'probability']))

    acc = accuracy_score(list(df['class']), list(df['prediction']))
    f1 = f1_score(list(df['class']), list(df['prediction']), average='weighted')
    cluster_acc = cluster_accuracy(df)
    ece = get_ece(df)
    lang_3 = accuracy_score(language_true_3, language_pred_3)
    lang_10 = accuracy_score(language_true_10, language_pred_10)
    lang_30 = accuracy_score(language_true_30, language_pred_30)
    cluster_wise_scores = get_cluster_wise(df)
    # eer = EER()
    print(f"Accuracy: {acc}")
    print(f"Weighted F1: {f1}")
    print(f"Cluster Accuracy: {cluster_acc}")
    print(f"ECE: {ece}")
    # print(f"EER: {eer}")
    print(f"Language 3/10/30 seconds: {lang_3}, {lang_10}, {lang_30}")
    print(f"Cluster wise scores: {cluster_wise_scores}\n------")
    return acc, f1, cluster_acc, ece, lang_3, lang_10, lang_30, cluster_wise_scores

print("Set 1")
a1, f1, c1, e1, l3_1, l10_1, l30_1, cluster_wise_1 = metrics(set_1)
print("Set 2")
a2, f2, c2, e2, l3_2, l10_2, l30_2, cluster_wise_2 = metrics(set_2)
print("Set 3")
a3, f3, c3, e3, l3_3, l10_3, l30_3, cluster_wise_3 = metrics(set_3)

avg_cluster_wise = (np.array(cluster_wise_1) + np.array(cluster_wise_2) + np.array(cluster_wise_3))/3

print("Average")
print(f"Accuracy: {(a1 + a2 + a3)/3}")
print(f"Weighted F1: {(f1 + f2 + f3)/3}")
print(f"Cluster Accuracy: {(c1 + c2 + c3)/3}")
print(f"ECE: {(e1 + e2 + e3)/3}")
print(f"Lang 3: {(l3_1 + l3_2 + l3_3)/3}, Lang 10: {(l10_1 + l10_2 + l10_3)/3} Lang 30: {((l30_1 + l30_2 + l30_3)/3)}")
print(f"Cluster wise scores: {avg_cluster_wise}")