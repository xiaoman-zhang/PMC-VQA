import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import PIL
import numpy as np
import torch.nn.functional as F
import transformers
import pandas as pd
import random
import copy
from .randaugment import RandomAugment    
from PIL import Image
import tqdm
aaa = 0        
class PMC_QA_Dataset(Dataset):
    def __init__(self,  img_dir, csv_path, tokenizer_path, img_tokens = 32, seq_length = 512,voc_size = 32000, mode = 'Train',start = 0,text_type = 'random',no_image = False):
        self.img_root = img_dir
        self.data = pd.read_csv(csv_path).iloc[start:]
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token_id=0
        self.tokenizer.eos_token_id=1
        self.img_padding = [-100 for i in range(img_tokens)]
        self.attn_padding = [1 for i in range(img_tokens)]
        self.H = 512
        self.W = 512
        self.C = 3
        self.text_type = text_type
        self.no_image = no_image
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop((self.H,self.W),scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ]) 
        if mode == 'Test':
            self.transform = transforms.Compose([                        
                transforms.Resize((self.H,self.W), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
            
            if self.text_type =='random':
                self.text_type = 'choice'
            
        self.mode = mode
        self.seq_length = seq_length
        self.voc_size = voc_size
        
    def __len__(self):
        return len(self.data)
    
    def random_answer(self, Question, choice_list,Answer,caption):
        p = random.random()
        Combined_choice = ''
        for choice in choice_list:
            Combined_choice = Combined_choice + choice
        if self.text_type =='random':                
            if p<=0.33:
                Answer = Answer.replace('A:','').replace('B:','').replace('C:','').replace('D:','')
                pre_text = 'Question: '+ Question +'The Answer is:'
                final_o = 'Question: '+ Question +'The Answer is:' + Answer
            if p>0.33 and p<=0.66:
                pre_text = 'Question: '+ Question + 'Choices:' + Combined_choice +'The Answer is:' 
                final_o = 'Question: '+ Question + 'Choices:' + Combined_choice +'The Answer is:' +Answer 
            if p>0.66:
                pre_text = ''
                final_o = caption
        if self.text_type =='caption':
            pre_text = ''
            final_o = caption    
        if self.text_type =='blank':
            Answer = Answer.replace('A:','').replace('B:','').replace('C:','').replace('D:','')
            pre_text = 'Question: '+ Question +'The Answer is:'
            final_o = 'Question: '+ Question +'The Answer is:' + Answer
        if self.text_type =='choice':
            pre_text = 'Question: '+ Question + 'Choices:' + Combined_choice +'The Answer is:' 
            final_o = 'Question: '+ Question + 'Choices:' + Combined_choice +'The Answer is:' +Answer 
        #answer_tokenized = self.tokenizer(text)

        return pre_text,final_o

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        Question  = sample['Question']
        Choice_A = sample['Choice A']
        Choice_B = sample['Choice B']
        Choice_C = sample['Choice C']
        Choice_D = sample['Choice D']
        caption = sample['Caption']
        choice_list = [Choice_A,Choice_B,Choice_C,Choice_D]
        Anwser = sample['Anwser']
        
        if not self.no_image:
        ##### read image pathes #####
            img_path = self.img_root + sample['Figure_path']
            img = PIL.Image.open(img_path).convert('RGB') 
            image = self.transform(img) 
        
        #Question_id = np.array(self.tokenizer(Question)['input_ids'])
        if self.mode == 'Train':
            pre_text,final_o = self.random_answer(Question,choice_list,Anwser,caption)
            
            final_o = self.tokenizer(final_o)
            input_ids = final_o['input_ids']
            input_ids.append(self.tokenizer.eos_token_id)
            input_ids = np.array(input_ids)
            
            if len(input_ids) < self.seq_length:
                input_ids = np.pad(input_ids, (0, self.seq_length - len(input_ids)), 'constant', constant_values=0)
            else:
                input_ids = input_ids[:self.seq_length]
                
            #attention = np.array(self.attn_padding + final_o['attention_mask'])
            label = copy.deepcopy(input_ids)
            label[label==0] = -100
            if pre_text != '':
                pre_text = self.tokenizer(pre_text)
                if len(pre_text['input_ids'])<len(label):
                    #label = np.array(label)
                    label[:len(pre_text['input_ids'])] = -100
            label = label.tolist()
            if not self.no_image:
                label = np.array(self.img_padding + label)
            
                item = {
                    'input_ids': input_ids,       
                    'images': image,
                    'labels': label,
                }
            else:
                label = np.array(label)
                item = {
                    'input_ids': input_ids, 
                    'labels': label,
                }
            return item
        
        if self.mode == 'Test':
            Combined_choice = ''
            #random.shuffle(choice_list)
            reflect = {0:' A:',1:' B:',2:' C:',3:' D:' }
            for i,choice in enumerate(choice_list):
                if Anwser == choice:
                    Anwser = Anwser.replace(' A:',reflect[i]).replace(' B:',reflect[i]).replace(' C:',reflect[i]).replace(' D:',reflect[i])
                if Choice_A == choice:
                    Choice_A = Choice_A.replace(' A:',reflect[i]).replace(' B:',reflect[i]).replace(' C:',reflect[i]).replace(' D:',reflect[i])
                if Choice_B == choice:
                    Choice_B = Choice_B.replace(' A:',reflect[i]).replace(' B:',reflect[i]).replace(' C:',reflect[i]).replace(' D:',reflect[i])
                if Choice_C == choice:
                    Choice_C = Choice_C.replace(' A:',reflect[i]).replace(' B:',reflect[i]).replace(' C:',reflect[i]).replace(' D:',reflect[i])
                if Choice_D == choice:
                    Choice_D = Choice_D.replace(' A:',reflect[i]).replace(' B:',reflect[i]).replace(' C:',reflect[i]).replace(' D:',reflect[i]) 
                Combined_choice = Combined_choice + choice.replace(' A:',reflect[i]).replace(' B:',reflect[i]).replace(' C:',reflect[i]).replace(' D:',reflect[i])
            if not self.no_image:
                item = {
                    'input_ids': 'Question: '+ Question + 'Choices:' + Combined_choice +'The Answer is:',
                    'img_path': sample['Figure_path'],       
                    'images': image,
                    'labels': Anwser,
                    'Choice_A': Choice_A,
                    'Choice_B': Choice_B,
                    'Choice_C': Choice_C,
                    'Choice_D': Choice_D,
                }
            else:
                item = {
                    'input_ids': 'Question: '+ Question + 'Choices:' + Combined_choice +'The Answer is:',
                    'img_path': sample['Figure_path'],       
                    'labels': Anwser,
                    'Choice_A': Choice_A,
                    'Choice_B': Choice_B,
                    'Choice_C': Choice_C,
                    'Choice_D': Choice_D,
                }
            if not self.no_image:
                if self.text_type=='blank':
                    item = {
                        'input_ids': 'Question: '+ Question + 'The Answer is:',
                        'img_path': sample['Figure_path'],       
                        'images': image,
                        'labels': Anwser,
                        'Choice_A': Choice_A,
                        'Choice_B': Choice_B,
                        'Choice_C': Choice_C,
                        'Choice_D': Choice_D,
                    }
            else:
                if self.text_type=='blank':
                    item = {
                        'input_ids': 'Question: '+ Question + 'The Answer is:',
                        'img_path': sample['Figure_path'],       
                        'labels': Anwser,
                        'Choice_A': Choice_A,
                        'Choice_B': Choice_B,
                        'Choice_C': Choice_C,
                        'Choice_D': Choice_D,
                    }
            
            return item
        

# img_dir = '/nvme/zhangruipeng/zhangxiaoman/data/PMC_OA_papers/figures/'
# csv_path = '/nvme/zhangruipeng/wuchaoyi/chatGPT_APIs/PMC_QA_project/Data/vqas/test_process.csv'
# tokenizer_dir = '/nvme/zhangruipeng/wuchaoyi/chatGPT_APIs/PMC_QA_project/LLAMA/tokenizer'
# dataset = PMC_QA_Dataset(img_dir,csv_path,tokenizer_dir)
# print(dataset[0])