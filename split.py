import pandas as pd
import argparse
import os
import sys

def split_excel_file(input_file, output_prefix="split", random_split=False, 
                     first_output=None, second_output=None):
    """
    Divide un file Excel in due metà uguali.
    
    Args:
        input_file (str): Percorso al file Excel da dividere
        output_prefix (str): Prefisso per i file di output
        random_split (bool): Se True, divide casualmente invece che sequenzialmente
        first_output (str): Nome personalizzato per il primo file di output
        second_output (str): Nome personalizzato per il secondo file di output
    
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
        
        # Calcola il punto medio
        midpoint = total_rows // 2
        
        if random_split:
            # Divisione casuale
            print("Esecuzione divisione casuale...")
            df = df.sample(frac=1).reset_index(drop=True)
        
        # Dividi il dataframe
        first_half = df.iloc[:midpoint].copy()
        second_half = df.iloc[midpoint:].copy()
        
        print(f"Prima metà: {len(first_half)} righe")
        print(f"Seconda metà: {len(second_half)} righe")
        
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
        print(f"Salvando la prima metà in {first_output_path}...")
        first_half.to_excel(first_output_path, index=False)
        
        print(f"Salvando la seconda metà in {second_output_path}...")
        second_half.to_excel(second_output_path, index=False)
        
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
    parser = argparse.ArgumentParser(description='Divide un file Excel in due metà uguali.')
    parser.add_argument('input_file', help='Percorso al file Excel da dividere')
    parser.add_argument('--prefix', default='split', help='Prefisso per i file di output')
    parser.add_argument('--random', action='store_true', help='Effettua una divisione casuale')
    parser.add_argument('--output1', help='Nome personalizzato per il primo file di output')
    parser.add_argument('--output2', help='Nome personalizzato per il secondo file di output')
    
    # Analizza gli argomenti
    args = parser.parse_args()
    
    # Esegui la divisione
    split_excel_file(args.input_file, args.prefix, args.random, args.output1, args.output2)

if __name__ == "__main__":
    main()