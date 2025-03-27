import pandas as pd
import argparse
import os
import sys

def split_excel_file(input_file, output_prefix="split", random_split=False, 
                     first_output=None, second_output=None, split_percentage=80):
    """
    Divide un file Excel in due parti secondo la percentuale specificata.
    Permuta sempre in modo casuale le righe prima della divisione.
    
    Args:
        input_file (str): Percorso al file Excel da dividere
        output_prefix (str): Prefisso per i file di output
        random_split (bool): Parametro mantenuto per retrocompatibilità (non ha effetto)
        first_output (str): Nome personalizzato per il primo file di output
        second_output (str): Nome personalizzato per il secondo file di output
        split_percentage (int): Percentuale per la prima parte (default: 80%)
    
    Returns:
        tuple: Percorsi dei due file creati
    """
    try:
        # Leggi il file Excel
        print(f"Leggendo il file {input_file}...")
        df = pd.read_excel(input_file)
        
        # Ottieni le informazioni sul file
        total_rows = len(df)
        print(f"Totale righe nel file: {total_rows}")
        
        # Permuta sempre casualmente le righe
        print("Permutazione casuale delle righe...")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calcola il punto di divisione basato sulla percentuale
        split_point = int(total_rows * split_percentage / 100)
        
        # Dividi il dataframe
        first_part = df.iloc[:split_point].copy()
        second_part = df.iloc[split_point:].copy()
        
        print(f"Prima parte ({split_percentage}%): {len(first_part)} righe")
        print(f"Seconda parte ({100-split_percentage}%): {len(second_part)} righe")
        
        # Crea i nomi dei file di output
        base_dir = os.path.dirname(input_file)
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        
        if first_output is None:
            first_output = f"{output_prefix}_{base_filename}_1.xlsx"
        
        if second_output is None:
            second_output = f"{output_prefix}_{base_filename}_2.xlsx"
        
        # Percorsi completi
        first_output_path = os.path.join(base_dir, first_output)
        second_output_path = os.path.join(base_dir, second_output)
        
        # Salva i file
        print(f"Salvando la prima parte ({split_percentage}%) in {first_output_path}...")
        first_part.to_excel(first_output_path, index=False)
        
        print(f"Salvando la seconda parte ({100-split_percentage}%) in {second_output_path}...")
        second_part.to_excel(second_output_path, index=False)
        
        print("Divisione completata con successo!")
        return first_output_path, second_output_path
        
    except FileNotFoundError:
        print(f"Errore: Il file {input_file} non è stato trovato.")
        return None, None
    except Exception as e:
        print(f"Errore durante la divisione del file: {str(e)}")
        return None, None

def main():
    # Configura il parser degli argomenti
    parser = argparse.ArgumentParser(description='Divide un file Excel in due parti secondo una percentuale specificata.')
    parser.add_argument('input_file', help='Percorso al file Excel da dividere')
    parser.add_argument('--prefix', default='split', help='Prefisso per i file di output')
    parser.add_argument('--random', action='store_true', help='Parametro mantenuto per retrocompatibilità (non ha effetto)')
    parser.add_argument('--output1', help='Nome personalizzato per il primo file di output')
    parser.add_argument('--output2', help='Nome personalizzato per il secondo file di output')
    parser.add_argument('--percentage', type=int, default=80, help='Percentuale per la prima parte (default: 80)')
    
    # Analizza gli argomenti
    args = parser.parse_args()
    
    # Esegui la divisione
    split_excel_file(args.input_file, args.prefix, args.random, args.output1, args.output2, args.percentage)

if __name__ == "__main__":
    main()