import pandas as pd
import matplotlib.pyplot as plt

def plot_training_vs_test(training_files, test_files):
    assert len(training_files) == len(test_files), "Il numero di file di training e test deve essere uguale."
    
    for i, (training_file, test_file) in enumerate(zip(training_files, test_files)):
        train_df = pd.read_csv(training_file)
        test_df = pd.read_csv(test_file)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'Session {i+1}')
        
        # Plotta la loss di training e test nello stesso grafico
        axes[0].plot(train_df['step'], train_df['loss'], label=f'Training Loss ({training_file})', marker='o', linestyle='--', color='blue')
        axes[0].plot(test_df['step'], test_df['loss'], label=f'Test Loss ({test_file})', marker='s', linestyle='-', color='red')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training vs Test Loss')
        axes[0].legend()
        axes[0].grid()
        
        # Plotta l'accuracy di test
        axes[1].plot(test_df['step'], test_df['accuracy'], label=f'Test Accuracy ({test_file})', marker='^', linestyle=':')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Test Accuracy')
        axes[1].legend()
        axes[1].grid()
        
        plt.tight_layout()
        plt.show()
plot_training_vs_test(['first_train.csv', 'second_train.csv', 'third_train.csv','fourth_train.csv','fifth_train.csv'],['first_eval.csv', 'second_eval.csv','third_eval.csv','fourth_eval.csv','fifth_eval.csv'])