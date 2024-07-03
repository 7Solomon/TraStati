import subprocess
import os

def ask_which_path(possible_policy_paths):
    print("Possible policy file paths:")
    for i, path in enumerate(possible_policy_paths):
        print(f'{i}: {path}')

    idx = input("Welcher Path sieht richtig aus? ")
    try:
        idx = int(idx)
        return possible_policy_paths[idx]
    except ValueError:
        print('Du hast kein Int angegeben, oder er war außerhalb des Bereichs')
        return None

def get_policy_file_path():
    try:
        # Use 'magick identify' instead of 'identify'
        output = subprocess.run(['magick', 'identify', '-list', 'policy'], capture_output=True, text=True)
    except FileNotFoundError:
        print("The 'magick' command was not found. Ensure ImageMagick is installed and in the system's PATH.")
        return None

    possible_policy_paths = []
    for line in output.stdout.split('\n'):
        if 'Path:' in line:
            possible_policy_paths.append(line.split(': ')[1].strip())
    
    if not possible_policy_paths:
        print("No policy paths found.")
        return None
    
    path = ask_which_path(possible_policy_paths)
    return path

def update_policy_file(policy_file_path):
    if not policy_file_path:
        print("No policy file path provided.")
        return

    try:
        with open(policy_file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("The policy file was not found.")
        return
    except PermissionError:
        print("You do not have the necessary permissions to read the policy file.")
        return

    try:
        with open(policy_file_path, 'w') as file:
            for line in lines:
                if '<policy domain="coder" rights="none" pattern="PDF" />' in line:
                    line = line.replace('<policy domain="coder" rights="none" pattern="PDF" />',
                                        '<policy domain="coder" rights="read | write" pattern="PDF" />')
                file.write(line)
        print('-----------')
        print("The policy file has been updated.")
        print('-----------')
    except PermissionError:
        print('-----------')
        print("You do not have the necessary permissions to edit the policy file.")
        print('-----------')
        return
    
import os
import subprocess
import winreg

def add_latex_search_path_permanent():
    relative_path = r"src\data_folder\get_system_image"
    # Konvertiere relativen Pfad in absoluten Pfad
    abs_path = os.path.abspath(relative_path)
    
    # Normalisiere den Pfad und ersetze Backslashes durch Forward Slashes
    new_path = os.path.normpath(abs_path).replace("\\", "/")
    
    try:
        # Öffne den Registry-Schlüssel für Umgebungsvariablen
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS)
        
        try:
            # Versuche, den aktuellen Wert von TEXINPUTS zu lesen
            current_texinputs, _ = winreg.QueryValueEx(key, "TEXINPUTS")
        except WindowsError:
            # Wenn TEXINPUTS nicht existiert, erstelle es
            current_texinputs = ""
        
        # Füge den neuen Pfad hinzu, wenn er noch nicht vorhanden ist
        if new_path not in current_texinputs:
            if current_texinputs:
                new_texinputs = f"{new_path};{current_texinputs}"
            else:
                new_texinputs = f"{new_path};"
            
            # Setze den neuen Wert in der Registry
            winreg.SetValueEx(key, "TEXINPUTS", 0, winreg.REG_EXPAND_SZ, new_texinputs)
            
            print(f"Pfad '{new_path}' wurde permanent zu TEXINPUTS hinzugefügt.")
            
            # Aktualisiere die Umgebungsvariablen für den aktuellen Prozess
            os.environ['TEXINPUTS'] = new_texinputs
            
            # Aktualisiere die TeX-Datenbank
            try:
                subprocess.run(['mktexlsr'], check=True)
                print("TeX-Datenbank wurde aktualisiert.")
            except subprocess.CalledProcessError:
                print("Warnung: Konnte die TeX-Datenbank nicht aktualisieren.")
            except FileNotFoundError:
                print("Warnung: mktexlsr nicht gefunden. TeX-Datenbank wurde nicht aktualisiert.")
        else:
            print(f"Pfad '{new_path}' ist bereits in TEXINPUTS enthalten.")
        
    except Exception as e:
        print(f"Fehler beim Hinzufügen des Pfads: {e}")
    finally:
        winreg.CloseKey(key)
    
    return os.environ.get('TEXINPUTS', '')



if __name__ == '__main__':
    if os.name == 'nt':  # Windows
        updated_texinputs = add_latex_search_path_permanent()
        print(f"Aktualisierte TEXINPUTS: {updated_texinputs}")
    else:
        policy_file_path = get_policy_file_path()
        if policy_file_path:
            print(f'Policy file, {policy_file_path} wird bearbeitet..')
            update_policy_file(policy_file_path)
    
