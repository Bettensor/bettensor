from bettensor.protocol import TeamGame

def custom_serializer(obj):
    if isinstance(obj, TeamGame):
        return obj.__dict__
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
