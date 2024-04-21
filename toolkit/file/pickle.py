import pickle

def read_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        raise FileNotFoundError("File not found. Please provide a valid file path.")
    except Exception as e:
        raise Exception("An error occurred: ", e)

def write_pickle_file(data, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print("Data has been successfully written to", file_path)
    except Exception as e:
        raise Exception("An error occurred: ", e)