import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    file_path = "../train_data/dontpatronizeme_pcl.tsv"
    df = pd.read_csv(file_path, sep='\t', header=None, names=['paragraph-id', 'keyword', 'countrycode', "paragraph", "label"])
    df_filtered = df[df['paragraph'].notna()]

    df_filtered['label'].hist()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Label')
    plt.show()