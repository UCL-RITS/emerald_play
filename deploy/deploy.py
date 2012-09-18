from config import *
from fabric.contrib.project import *
from fabric.contrib.files import *
from functools import wraps
import time
import re
@task
def stat():
	return run('bjobs')
	
@task
def sync():
	rsync_project(remote_dir=env.remote_source_path,local_dir=env.local_source_path,exclude=env.exclude)
	
@task
def wipe():
	run("rm -rf {remote_source_path}".format(**env))
	
def eachproject(task):
	@wraps(task)
	def wrapper(*args,**kwds):
		# if an explicit project is given as a kwarg or pos arg, just yield
		if len(args)>0:
			env.project=args[0]
			return task(*args,**kwds)
		elif 'project' in kwds:
			env.project=args[0]
			return task(*args,**kwds)
		else:
			projects=kwds.get('projects',env.projects)
			results=[]
			for project in projects:
				env.project=project
				kwds.update(project=project)
				results.append(task(*args,**kwds))
			return results
	return wrapper

@task
@eachproject
def cold(project):
	execute(clear_build,project)
	execute(sync)
	execute(configure,project)
	execute(make,project)
	execute(install,project)

@task
@eachproject
def build(project):
	execute(sync)
	execute(configure,project)
	execute(make,project)
	execute(install,project)
	
@task(alias="inc")
@eachproject
def incremental(project):
	execute(sync)
	execute(make,project)
	execute(install,project)

@task(alias="clear")
@eachproject
def clear_build(project):
	run("rm -rf {remote_build_path}/{project}".format(**env))

@task(alias="config")
@eachproject
def configure(project):
	run("mkdir -p {remote_build_path}/{project}".format(**env))
	with prefix("module load intel cuda"):
		with cd("{remote_build_path}/{project}".format(**env)):
			run("~/bin/cmake ~/{remote_source_path}/{project} -DCMAKE_INSTALL_PREFIX=~/{remote_install_path}/{project}".format(**env))


@task
@eachproject
def make(project):
	with prefix("module load intel cuda"):
		with cd("{remote_build_path}/{project}".format(**env)):
			run("make")

@task(alias="config")
@eachproject
def install(project):
	with prefix("module load intel cuda"):
		with cd("{remote_build_path}/{project}".format(**env)):
			run("make install")

@task
@eachproject
def sub(project):
	run("mkdir -p ~/{jobscripts}/".format(**env))
	run("mkdir -p ~/{remote_results_path}/{project}".format(**env))
	upload_template(filename="{templates}/default.sh".format(**env),destination="~/{jobscripts}/".format(**env),context=env)
	with prefix("module load intel cuda"):
		run("bsub < ~/{jobscripts}/default.sh".format(**env))

@task
def wait():
  """Wait until all jobs currently qsubbed are complete, then return"""
  while not re.search("No unfinished",stat()):
	time.sleep(10)

@task
@eachproject
def latest(project):
	latest=local("ls -1tr {localroot}/{project}/results".format(**env),capture=True)
	last=latest.split('\n')[-3:-1]
	for file in last:
		local("cat {localroot}/{project}/results/{file}".format(file=file,**env))

@task
@eachproject
def fetch(project):
	local("mkdir -p {localroot}/{project}/results".format(**env))
	local("rsync -pthrvz {username}@{remote}:{remote_results_path}/{project}/ {localroot}/{project}/results".format(**env))