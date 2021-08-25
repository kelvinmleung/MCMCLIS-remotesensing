# Aug 24, 2021
# Use surface_model.py to generate a surface model


import sys
sys.path.insert(0, '../isofit/isofit/utils/')

from surface_model import surface_model

config_path = 'setup/ang20140612/config/ang20140612_surface.json'

surface_model(config_path)

