
import tqdm
import torch
class Logger:
    
    def __init__(self, progress_bar:tqdm, ema_weight:float=0.6):
        self.pbar = progress_bar
        self.ema_weight = ema_weight

        self.log_dic = {}

    def update(self, display_dict:dict):
        
        for key, (newval,logtype,fmt) in display_dict.items():
            if torch.is_tensor(newval):
                newval = newval.item()
            if logtype.strip().lower() == "ema":    
                self.log_dic[key] = (self.log_dic.get(key,(0,fmt))[0] * self.ema_weight + newval * (1.0-self.ema_weight),fmt)
            elif logtype.strip().lower() == "update":
                self.log_dic[key] = (newval,fmt)
            else:
                raise NotImplementedError

    def show(self):
        self.pbar.set_postfix({k:format(val,fmt) for k,(val,fmt) in self.log_dic.items()})
        self.pbar.update(10)