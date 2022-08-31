from kfp.components import load_component_from_text

def serve():
    """ Return bentoml serving container info """
    return load_component_from_text("""
name: Push model to bentoml
implementation:
  container:
    image: jerife/bentoml_serve:v0.6
""")