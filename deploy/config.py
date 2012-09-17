import yaml
import os
from fabric.api import *

env.localroot=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
configuration=yaml.load(open(os.path.join(env.localroot,'deploy','config.yml')))
user_configuration=yaml.load(open(os.path.join(env.localroot,'deploy','config_user.yml')))
configuration.update(user_configuration)
env.update(configuration)
env.hosts=['%s@%s'%(env.username,env.remote)]
env.local_source_path=env.localroot+'/'
env.templates=os.path.join(env.localroot,'deploy','templates')