import sys
import argparse

class Config:

    def __init__(self):
        super(Config,self).__init__()
        self.parser = argparse.ArgumentParser(description='dogvscat')
        self.config = dict()
        self._add_common_setting()
        self.args = self.parser.parse_args()
        self._load_common_setting()
        


    def _add_common_setting(self):
     self.parser.add_argument('--num_workers',type=int,default=12)
     self.parser.add_argument('--batchSize',type=int,default=32)
     self.parser.add_argument('--nepoch',type=int,default=20)
     self.parser.add_argument('--lr',type=float,default=0.001)
     self.parser.add_argument('--cuda',type=str,default='0')
     self.parser.add_argument('--optimizer',type=str,default='Adam')


     
    def _load_common_setting(self):
     self.config['num_workers'] = self.args.num_workers
     self.config['batchSize'] = self.args.batchSize
     self.config['nepoch'] = self.args.nepoch
     self.config['lr'] = self.args.lr
     self.config['cuda'] = self.args.cuda
     self.config['optimizer'] = self.args.optimizer
    
    def print_config(self,_print=None):
        _print("==================== basic setting start ====================") 
        for arg in self.config:
            _print('{:20}:{}'.format(arg,self.config[arg]))

        _print("==================== basic setting start ====================")

    def get_config(self):
        return self.config   
