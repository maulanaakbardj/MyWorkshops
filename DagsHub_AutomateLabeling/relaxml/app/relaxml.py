import logging
import os
import requests

from label_studio_tools.core.label_config import parse_config
from requests.auth import HTTPBasicAuth
from typing import List

from .datamodel import Setup, Task
from .utils import uri_to_url, download_url


class RelaxML:
    def __init__(self):
        '''Good place to load your model and setup variables'''

        self.project = None
        self.schema = None
        self.hostname = None
        self.access_token = None
 
        self.user = os.getenv("DAGSHUB_USER_NAME")
        self.token = os.getenv("DAGSHUB_TOKEN")
        self.repo = os.getenv("DAGSHUB_REPO_NAME")
        self.owner = os.getenv("DAGSHUB_REPO_OWNER")

        # HERE: Load model


    def setup(self, setup: Setup):
        '''Store the setup information sent by Label Studio to the ML backend'''

        self.project = setup.project
        self.parsed_label_config = parse_config(setup.label_schema)
        self.hostname = setup.hostname
        self.access_token = setup.access_token

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']

    
    def send_predictions(self, result):
        '''Send prediction results to Label Studio'''

        url = f'https://dagshub.com/{self.owner}/{self.repo}/annotations/api/predictions/'
        auth = HTTPBasicAuth(self.user, self.token)
        res = requests.post(url, auth=auth, json=result)
        if res.status_code != 200:
            logging.warning(res)
      

    def predict(self, tasks: List[Task]):
        '''Add predict logic here and call `send_predictions` for each task'''

        pass