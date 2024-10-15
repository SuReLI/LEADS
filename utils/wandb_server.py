import wandb
import os
from pathlib import Path
class WandbServer:
	def __init__(self,project_name,name=None, config={}) -> None:
		# wandb ENV variables
		# self.hostname=os.environ['HOSTNAME']
		# if 'armos' in self.hostname:
		# 	os.environ["WANDB_API_KEY"] = ""
		# 	os.environ["WANDB_BASE_URL"] = ""
		# 	# os.environ["WANDB_MODE"] = "dryrun"
		# 	os.environ["WANDB_DIR"] = "../"
		# else : 
		os.environ["WANDB_API_KEY"] = ""
		os.environ["WANDB_MODE"] = "dryrun"
		os.environ["WANDB_DIR"] = "../"
		# init
		wandb.init(project=project_name,config=config)
		wandb.run.name=name if name !=None else wandb.run.name
	
	