from config import *
from fabric.contrib.project import *
from fabric.contrib.files import *
@task
def stat():
	run('bjobs')
	
@task
def sync():
	rsync_project(remote_dir=env.remote_source_path,local_dir=env.local_source_path,exclude=env.exclude)
	
@task
def wipe():
	run("rm -rf {remote_source_path}".format(**env))
	
@task
def build(project):
	env.project=project
	run("rm -rf {remote_build_path}/{project}".format(**env))
	run("mkdir -p {remote_build_path}/{project}".format(**env))
	with prefix("module load intel cuda"):
		with cd("{remote_build_path}/{project}".format(**env)):
			run("~/bin/cmake ~/{remote_source_path}/{project} -DCMAKE_INSTALL_PREFIX=~/{remote_install_path}/{project}".format(**env))
			run("make")
			run("make install")

@task
def sub(project):
	env.project=project
	run("mkdir -p ~/{jobscripts}/".format(**env))
	run("mkdir -p ~/{results}/{project}".format(**env))
	upload_template(filename="{templates}/{project}.sh".format(**env),destination="~/{jobscripts}/".format(**env),context=env)
	with prefix("module load intel cuda"):
		run("bsub ~/{jobscripts}/{project}.sh".format(**env))