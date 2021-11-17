import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

mpl.rcParams['figure.subplot.top'] = 0.50
mpl.rcParams['figure.subplot.bottom'] = 0.44
mpl.rcParams['figure.subplot.left'] = 0.01
mpl.rcParams['figure.subplot.right'] = 0.995

#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "DejaVu Sans"
#sns.set(font="Times New Roman")

def store_heat(df_data, save_dir):
       cmap = sns.light_palette((260, 75, 60), input="husl")
       sns.heatmap(df_data, annot=True, yticklabels=False, cmap=cmap, cbar=False)
       plt.savefig(save_dir,bbox_inches='tight')
       plt.close()

#一个使用例子，df_data为DataFrame类型的数据，列名（columns需设置为对应显示的字符串）
#path为保存路径
df_data = pd.DataFrame([[0.211079491893512, 0.15495145845407637, 0.20774342826457778, 0.42599948790355846, 0.18637215614613584, 0.7699411449202395, 0.2410692465387192, 0.5193177575044197, 1.0, 0.26908530253983787, 0.26385087160798687, 0.921529305614449, 0.12745411350557215]],columns=["All","of","the","apetizers","are","good","and","the","Sangria","is","very","good","."])
path = "apetizers_LSTM_test.pdf"

store_heat(df_data,path)
