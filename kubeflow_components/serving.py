from kfp.components import load_component_from_text

def serve():
    """ Return bentoml serving container info """
    return load_component_from_text("""
name: Custom_Plugin_1
description: This is an example
implementation:
  container:
    image: hello-world                           
""")