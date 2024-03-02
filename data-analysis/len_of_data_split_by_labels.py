
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    file_path = "../train_data/dontpatronizeme_pcl.tsv"
    df = pd.read_csv(file_path, sep='\t', header=None, names=['paragraph-id', 'keyword', 'countrycode', "paragraph", "label"])
    df_filtered = df[df['paragraph'].notna()]
    df_filtered['paragraph_length'] = df_filtered['paragraph'].apply(lambda x: len(x.split()))

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_filtered, x='paragraph_length', hue='label', element='step', palette='viridis', bins=30,
                 kde=True)
    plt.xlabel('Paragraph Length (Number of Words)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Paragraph Lengths by Label')
    plt.show()