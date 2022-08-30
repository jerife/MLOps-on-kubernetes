import yaml
def yaml_parser(yaml_path):
    with open(yaml_path) as file_reader:
        yaml_info = yaml.load(file_reader, Loader=yaml.FullLoader)
    return yaml_info