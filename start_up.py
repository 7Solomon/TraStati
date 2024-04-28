import subprocess

def ask_which_path(possible_policy_paths):
    print("Possible policy file paths:")
    for i, path in enumerate(possible_policy_paths):
        print(f'{i}: {path}')

    idx = input("Welcher Path sieht richtig aus? ")
    try:  
        idx = int(idx)
        return possible_policy_paths[idx]
    except:
        print('Du hast kein Int angegeben, oder er war auserhalb des Bereichs')


def get_policy_file_path():
    output = subprocess.run(['identify', '-list', 'policy'], capture_output=True, text=True)
    
    possible_policy_paths = [None]
    for line in output.stdout.split('\n'):
        if 'Path:' in line:
            possible_policy_paths.append(line.split(': ')[1].strip())
    
    path =  ask_which_path(possible_policy_paths)
    return path
        
     

    
def update_policy_file(policy_file_path):
    #policy_file_path = "/etc/ImageMagick-6/policy.xml"

    try:
        # Read the contents of the policy file
        with open(policy_file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("The policy file was not found.")
        return
    
    try:
        # Uncomment or add the necessary line for PDF conversion
        with open(policy_file_path, 'w') as file:
            for line in lines:
                if '<policy domain="coder" rights="none" pattern="PDF" />' in line:
                    line = line.replace('<policy domain="coder" rights="none" pattern="PDF" />',
                                    '<policy domain="coder" rights="read | write" pattern="PDF" />')
                file.write(line)
    except PermissionError as e:
        print('-----------')
        print(e)
        print('-----------')
        print("You do not have the necessary permissions to edit the policy file.")
        return

if __name__ == '__main__':
    policy_file_path = get_policy_file_path()
    print(f'Policy file, {policy_file_path} wird bearbeitet..')
    update_policy_file(policy_file_path)