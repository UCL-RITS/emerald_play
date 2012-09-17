from config import *
from fabric.contrib.project import *

@task
def stat():
	run('bjobs')
	
@task
def sync():
	rsync_project(remote_dir=env.remote_source_path,local_dir=env.local_source_path,exclude=env.exclude)
	
@task
def wipe():
	run("rm -rf {remote_source_path}".format(**env))