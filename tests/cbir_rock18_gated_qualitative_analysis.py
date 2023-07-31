import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glob import glob

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from PIL import Image

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

import dataloader.dataset_cbir as dataset
from src.model import model_siamese_rock18_gated as model_define
from src import parser_define


output_size = 512
train_args = parser_define.train_args_define()
db_image_path = train_args.db_path # image_paths should be fixed 
image_paths = glob(train_args.db_path+'/*.png') # put all imgs in the path into a list
metadata_path = train_args.metadata_path # metadata_path


def image_name_extractor(image_path_filename):
    # Remove unnecessary parts of image name
    counter_underscore = 0
    for l in range(len(image_path_filename)):
        if (image_path_filename[l] == "_") :
            counter_underscore += 1
            if (counter_underscore == 1):
                rocksample_name = image_path_filename[:l]
                # print (rocksample_name)
                break
    
    return rocksample_name


def ap_k(actual, predicted, k):
    relevant_counter = 0
    precisions = np.zeros(k)
    
    for i in range(len(predicted)):
        if (relevant_counter < k): 
            if (predicted[i] == actual) :
                relevant_counter += 1
                precisions[relevant_counter-1]=float(relevant_counter)/(i+1)
                
    ap = np.mean(precisions)

    return ap


def inference(model_resnet, model_rock, model_reg, test_loader, args):
    print('running inference for database')
    model_resnet.eval()
    model_rock.eval()
    model_reg.eval()

    predictions = np.zeros((len(test_loader), output_size))
    predictions_reg = np.zeros((len(test_loader), 2))
    
    for i, data in enumerate(test_loader, 0):
        # print ('CBIRDatabase: Computing prediction for {}th image'.format(i+1))
        img, label = data
        img = img.to(args.device)

        output = model_resnet(img)
        output_rock = model_rock(img)
        output_reg = model_reg(output_rock) #[Porosity, log(permeability) :natural logarithm
        
        # Normalization
        output = F.normalize(output, dim=1)
        output_rock = F.normalize(output_rock, dim=1)

        output = (output + output_rock)/2
            
        # Save prediction to list
        predictions[i] = torch.reshape(output, (output_size, )).cpu().detach().numpy()
        predictions_reg[i] = torch.reshape(output_reg, (2, )).cpu().detach().numpy()

    return predictions, predictions_reg


def query(model_resnet, model_rock, model_reg, img, num, predictions, actual):
    model_resnet.eval()
    model_rock.eval()
    model_reg.eval()

    # Utilize the neural network to the query image
    output = model_resnet(img)
    output_rock = model_rock(img)
    output_reg = model_reg(output_rock)

    # Normalization
    output = F.normalize(output, dim=1)
    output_rock = F.normalize(output_rock, dim=1)

    prediction = (output + output_rock)/2

    # print(prediction.shape)
    prediction = torch.reshape(prediction, (-1, 1)).cpu().detach().numpy().flatten()
    prediction_reg = torch.reshape(output_reg, (-1, 2)).cpu().detach().numpy().flatten()

    index = faiss.IndexFlatL2(output_size)
    predictions = predictions.astype('float32')
    index.add(predictions)

    distance, indices = index.search(prediction.reshape(1, output_size), num)

    imgIDs = indices[0]
    scores = np.sqrt(distance[0] / output_size)

    predicted = []

    for i in imgIDs:    
        img_path = pathlib.Path(image_paths[i])
        predicted.append(img_path.stem[:2])

    # MAP@10
    apk = ap_k(actual, predicted, 10)
    # print ("Average Precision@n:", apk)

    return imgIDs, scores, prediction_reg, apk


#1st row : top-3 retrieval from same sample, 2nd row : top-3 retrieval from same reservoir, 3rd row : top-3 retrieval from different reservoir, last row: bottom-3
def showResults(imgIDs, regs):
    fig, axs = plt.subplots(4, 3, constrained_layout=True)

    query_filename = pathlib.Path(image_paths[imgIDs[0]]).stem

    df_metadata = pd.read_csv(metadata_path)

    same_sample = []
    same_res = []
    diff_res = []
    most_diss_res = [imgIDs[-3], imgIDs[-2], imgIDs[-1]]
    abs_rank_sample = []
    abs_rank_res = []
    abs_rank_diff_res = []
    abs_rank_most_diss_res = [len(imgIDs)-2, len(imgIDs)-1, len(imgIDs)]

    sample_query = query_filename[:6]
    res_query = query_filename[:3]
    for i in range(len(imgIDs)):
        if pathlib.Path(image_paths[imgIDs[i]]).stem[:6] == sample_query:
            same_sample.append(imgIDs[i])
            abs_rank_sample.append(i)

    for i in range(len(imgIDs)):       
        if pathlib.Path(image_paths[imgIDs[i]]).stem[:3] == res_query:
            if pathlib.Path(image_paths[imgIDs[i]]).stem[:6] != sample_query:
                same_res.append(imgIDs[i])
                abs_rank_res.append(i)
        else:
            diff_res.append(imgIDs[i])      
            abs_rank_diff_res.append(i)

    if len(same_res)==0:
        same_res = same_sample
        abs_rank_res = abs_rank_sample

    same_res = [x for _,x in sorted(zip(abs_rank_res,same_res))]
    abs_rank_res = sorted(abs_rank_res)

    diff_res = [x for _,x in sorted(zip(abs_rank_diff_res,diff_res))]
    abs_rank_diff_res = sorted(abs_rank_diff_res)

    list_index = same_sample[0:3] + same_res[0:3] + diff_res[0:3] + most_diss_res # [0,1,2,3,4,5,6,7,8,-3,-2,-1]
    rank_index = abs_rank_sample[0:3] + abs_rank_res[0:3] + abs_rank_diff_res[0:3] + abs_rank_most_diss_res
    
    print("list for 4x3 results:", list_index)
    print("ranking for 4x3 results:", rank_index)

    for j in range(0, 4):
        for k in range(0, 3):
            index_image = list_index[3*j+k]
            img = Image.open(image_paths[index_image]).convert('LA')
            axs[j, k].imshow(img)

            found_microct_df = df_metadata[df_metadata['Image'].str.match(image_name_extractor(pathlib.Path(image_paths[index_image]).stem))]

            por = found_microct_df['Porosity'].values
            perm = found_microct_df['Permeability'].values

            axs[j, k].set_title(
                "{}\npor:{}, predicted_por:[{:.3f}], \n perm:{}, predicted_perm:[{:.3f}], \n ABS ranking:{}/{}".format(pathlib.Path(image_paths[index_image]).stem, por, regs[index_image][0],
                                                                    perm, np.power(10, 5*regs[index_image][1]), 1+rank_index[3*j + k], len(imgIDs)), pad=-20, fontsize=8)
            
            axs[j, k].axis('off')

    plt.show()


    
def main_func():
    train_args = parser_define.train_args_define()
    
    train_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('current device', train_args.device)

    # Setting a MicroCT dataset
    microct_dataset = dataset.Microct_Dataset(image_paths=db_image_path, 
                                            transform=transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.CenterCrop((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,))
                                                ]))

    test_loader = torch.utils.data.DataLoader(
        microct_dataset,
        batch_size=1, 
        shuffle=False)

    train_args.test_loader = test_loader

    print("loading pretrained ResNets")
    # Load resnet architectures
    model_resnet = models.resnet18(pretrained=True) 
    model_resnet = torch.nn.Sequential(*(list(model_resnet.children())[:-1])) # Remove last fc layer
    model_resnet.to(train_args.device)

    model_rock = models.resnet18(pretrained=True) 
    model_rock = torch.nn.Sequential(*(list(model_rock.children())[:-1])) # Remove last fc layer
    model_rock.to(train_args.device)
 
    # A regression model for rock
    model_reg = model_define.Regression_rock()
    model_reg.to(train_args.device)

    epochs_list = [920] 
    for j in epochs_list:
        model_dir = os.getcwd() + '/checkpoints/' # path 

        # Load a checkpoint for image
        # original saved file with DDP
        state_dict = torch.load(model_dir + 'epoch-{}.pt'.format(j)) # Resnet for image
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model_resnet.load_state_dict(new_state_dict)

        # Load a checkpoint for rock
        # original saved file with DDP
        state_dict = torch.load(model_dir + 'rock_epoch-{}.pt'.format(j))
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model_rock.load_state_dict(new_state_dict)

        # Load a checkpoint for rock
        # original saved file with DDP
        state_dict = torch.load(model_dir + 'reg_epoch-{}.pt'.format(j))
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model_reg.load_state_dict(new_state_dict)

        # Test image index from db
        index_image = 1209

        predictions, predictions_reg = inference(model_resnet, model_rock, model_reg, train_args.test_loader, train_args)

        # Query function
        num = len(image_paths)
        apks = np.zeros(len(test_loader))
        for i, data in enumerate(test_loader, 0):
            # print ('CBIRDatabaseClusters: Computing prediction for {}th query image'.format(i+1))
            if (i == index_image):
                img, label = data
                input_img = img.to(train_args.device)
                actual = pathlib.Path(image_paths[i]).stem[:2] 
                imgIDs, scores, regs, apks[i] = query(model_resnet, model_rock, model_reg, input_img, num, predictions, actual)
                img = Image.open(image_paths[index_image]).convert('LA') 
                showResults(imgIDs, predictions_reg) 
                break 
               

if __name__ == '__main__':
    main_func()
