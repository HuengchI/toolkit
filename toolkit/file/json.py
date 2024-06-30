import json
from typing import List, Dict, Union

def write_data_to_json_file(data: Union[List|Dict], filepath):
    try:
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)
        print(f'Data successfully written to {filepath}')
    except Exception as e:
        print(f'Error writing data to {filepath}: {e}')
    return filepath