
import os
import toml

def recursive_load(filename):

  if not os.path.isfile(filename):
    return {}

  # Load toml file
  config = toml.load(filename)

  # Check for linked toml files within
  if "include_configs" in config.keys():

    # Go through each linked toml file
    for include_filename in config["include_configs"]:
      include_config = recursive_load(include_filename)

        # Collate all arguments
      for key,value in include_config.items():
        if key not in config.keys() and key is not "include_configs":
          config[key] = value

  return config

