from collections import OrderedDict
import pandas as pd
import os
import matplotlib; matplotlib.use('Agg')
import torch
from shutil import copy
import numpy as np
import pandas as pd
import Custome_train2, Custome_train3
from spirl.components.params import get_args
from spirl.utils.general_utils import AttrDict
from spirl.components.checkpointer import load_by_key, freeze_modules
from torch import autograd
import time
from spirl.utils.general_utils import RecursiveAverageMeter, map_dict
from spirl.utils.eval_utils import ssim, psnr, mse

#init_path= "/home/workspace/skill/experiments/skill_prior_learning/kitchen/hierarchical_cl/normal-learning4/weights/weights_ep999.pth"
#init_path = "/home/workspace/skill/experiments/skill_prior_learning/kitchen/non-iid/weights/round-1000-weights.npz"


def pytorch_model_load(config,init_path):
    basemodel =  Custome_train2.ModelTrainer(args=config,cid=0,data_dir="./data7/kitchen/",grap_round=0)
    basemodel.model.load_state_dict(torch.load(init_path)['state_dict'], strict=False)
    return basemodel

def gl_numpy_model_load(config,init_path):
    basemodel =  Custome_train2.ModelTrainer(args=config,cid=0,data_dir="./data8/kitchen/",grap_round=0)
    device = torch.device("cuda")
    np_dict = np.load(init_path,allow_pickle=True)
    key_value = basemodel.model.state_dict().keys()
    params_dict = zip(key_value,np_dict)
    state_dict = OrderedDict()
    for k, v in params_dict:
        state_dict[k] = torch.Tensor(np_dict[v]).to(device)
    np_dict.close()
    l = []
    for d in state_dict :
        if "num_batches_tracked" in d :
            l.append(d)
    for d in l :
        del(state_dict[d])
    basemodel.model.load_state_dict(state_dict,strict = True)
    return basemodel

def ll_numpy_model_load(config,init_path):
    basemodel =  Custome_train2.ModelTrainer(args=config,cid=0,data_dir="./data/kitchen/",grap_round=0)
    device = torch.device("cuda")
    np_dict = np.load(init_path,allow_pickle=True)
    key_value = basemodel.model.state_dict().keys()
    params_dict = zip(key_value, np_dict["arr_0"])
    state_dict = OrderedDict()
    for k, v in params_dict:
        state_dict[k] = torch.Tensor(v).to(device)
    np_dict.close()
    l = []
    for d in state_dict :
        if "num_batches_tracked" in d :
            l.append(d)
    for d in l :
        del(state_dict[d])
    basemodel.model.load_state_dict(state_dict,strict = True)
    return basemodel

def ll_numpy_model_load_change(config,prior_path,init_path):
    change = AttrDict(
        path = "spirl/configs/skill_prior_learning/kitchen/FL_hierarchial_cl",
        prefix = "minner4",
        new_dir = False,
        dont_save = True,
        resume = "",
        train = True,
        test_prediction = True,
        skip_first_val = False,
        val_sweep = False,
        gpu = -1,
        strict_weight_loading = True,
        deterministic = False,
        log_interval = 500,
        per_epoch_img_logs = 1,
        val_data_size = 160,
        val_interval = 5,
        detect_anomaly = False,
        feed_random_data = False,
        train_loop_pdb = False,
        debug = False,
        save2mp4 = False
    )
    basemodel =  Custome_train2.ModelTrainer(args=config,cid=0,data_dir="./data/kitchen/",grap_round=0)
    basemodel.model.load_state_dict(torch.load(init_path)['state_dict'])
    chagnge_model =  Custome_train2.ModelTrainer(args=change,cid=0,data_dir="./data/kitchen/",grap_round=0)
    device = torch.device("cuda")
    np_dict = np.load(init_path,allow_pickle=True)
    pr_dict = np.load(prior_path)
    key_value = basemodel.model.state_dict().keys()
    params_dict = zip(key_value, np_dict["arr_0"], pr_dict)
    state_dict = OrderedDict()
    for k, v ,p in params_dict:
        if "p.0" in k :
            state_dict[k] = torch.Tensor(pr_dict[p]).to(device)
        else:
            state_dict[k] = torch.Tensor(v).to(device)
    np_dict.close()
    l = []
    for d in state_dict :
        if "num_batches_tracked" in d :
            l.append(d)
        if "classifier" in d :
            l.append(d)
    for d in l :
        del(state_dict[d])
    chagnge_model.model.load_state_dict(state_dict,strict = True)
    return chagnge_model

def gl_numpy_model_load_change(config,init_path):
    change = AttrDict(
        path = "spirl/configs/skill_prior_learning/kitchen/FL_hierarchial_cl",
        prefix = "minner4",
        new_dir = False,
        dont_save = True,
        resume = "",
        train = True,
        test_prediction = True,
        skip_first_val = False,
        val_sweep = False,
        gpu = -1,
        strict_weight_loading = True,
        deterministic = False,
        log_interval = 500,
        per_epoch_img_logs = 1,
        val_data_size = 160,
        val_interval = 5,
        detect_anomaly = False,
        feed_random_data = False,
        train_loop_pdb = False,
        debug = False,
        save2mp4 = False
    )
    basemodel =  Custome_train2.ModelTrainer(args=config,cid=0,data_dir="./data/kitchen/",grap_round=0)
    chagnge_model =  Custome_train2.ModelTrainer(args=change,cid=0,data_dir="./data/kitchen/",grap_round=0)
    device = torch.device("cuda")
    np_dict = np.load(init_path,allow_pickle=True)
    key_value = basemodel.model.state_dict().keys()
    params_dict = zip(key_value, np_dict)
    state_dict = OrderedDict()
    for k, v in params_dict:
        state_dict[k] = torch.Tensor(np_dict[v]).to(device)
    np_dict.close()
    l = []
    for d in state_dict :
        if "num_batches_tracked" in d :
            l.append(d)
        if "classifier" in d :
            l.append(d)
    for d in l :
        del(state_dict[d])
    chagnge_model.model.load_state_dict(state_dict,strict = True)
    return chagnge_model

def gl_torch_model_load_change(config,init_path):
    change = AttrDict(
        path = "spirl/configs/skill_prior_learning/kitchen/FL_hierarchial_cl",
        prefix = "minner4",
        new_dir = False,
        dont_save = True,
        resume = "",
        train = True,
        test_prediction = True,
        skip_first_val = False,
        val_sweep = False,
        gpu = -1,
        strict_weight_loading = True,
        deterministic = False,
        log_interval = 500,
        per_epoch_img_logs = 1,
        val_data_size = 160,
        val_interval = 5,
        detect_anomaly = False,
        feed_random_data = False,
        train_loop_pdb = False,
        debug = False,
        save2mp4 = False
    )
    basemodel =  Custome_train2.ModelTrainer(args=config,cid=0,data_dir="./data/kitchen/",grap_round=0)
    chagnge_model =  Custome_train2.ModelTrainer(args=change,cid=0,data_dir="./data/kitchen/",grap_round=0)
    device = torch.device("cuda")
    basemodel.model.load_state_dict(torch.load(init_path)['state_dict'])
    state_dict = basemodel.model.state_dict()
    l = []
    for d in state_dict :
        if "classifier" in d :
            l.append(d)
    for f in l :
        del(state_dict[f])
    chagnge_model.model.load_state_dict(state_dict,strict = True)
    return chagnge_model


def model_val(eval_models):
    for target_model in eval_models:
        target_model.model_test.load_state_dict(target_model.model.state_dict())
        target_model.model_test.eval()
    dataset_class = eval_models[0].conf.data.dataset_spec.dataset_class
    phase = 'val'
    total_score = []
    for i in range(7):
        score  = []
        data_dir =os.path.join(eval_models[0]._hp.data_dir, 'FL_{}'.format(i))
        val_loader = dataset_class(data_dir, eval_models[0].conf.data, resolution=eval_models[0].model.resolution,
                            phase=phase, shuffle=phase == "train", dataset_size=160). \
            get_data_loader(eval_models[0]._hp.batch_size, 1)
        for target_model in eval_models:
            target_model.evaluator.reset()
        print('Running Testing')
        with autograd.no_grad():
            for sample_batched in val_loader:
                    inputs = AttrDict(map_dict(lambda x: x.to(eval_models[0].device), sample_batched))
                    print(inputs.tasks)
                    for target_model in eval_models:
                        with target_model.model_test.val_mode():
                            target_model.evaluator.eval(inputs, target_model.model_test)
                            score.append(target_model.evaluator.dump_results(target_model.global_step))
                        output = target_model.model_test(inputs)
                        losses = target_model.model_test.loss(output, inputs)
                        print(losses)
        total_score.append(score)
        break
    return total_score

    
def get_z_space(eval_models):
    for target_model in eval_models:
        target_model.model_test.load_state_dict(target_model.model.state_dict())
        target_model.model_test.eval()
    dataset_class = eval_models[0].conf.data.dataset_spec.dataset_class
    phase = 'val'
    for i in range(7):
        data_dir =os.path.join(eval_models[0]._hp.data_dir, 'FL_{}'.format(i))
        val_loader = dataset_class(data_dir, eval_models[0].conf.data, resolution=eval_models[0].model.resolution,
                            phase=phase, shuffle=phase == "train", dataset_size=160). \
            get_data_loader(eval_models[0]._hp.batch_size, 1)
        for target_model in eval_models:
            target_model.evaluator.reset()
        print('Running Testing')
        with autograd.no_grad():
            output_list = []
            for sample_batched in val_loader:
                    inputs = AttrDict(map_dict(lambda x: x.to(eval_models[0].device), sample_batched))
                    for target_model in eval_models:
                        output_list.append(target_model.model_test(inputs))
                    #print(mse(output_list[0].reconstruction,(output_list[0].q_reconstruction )))
                    #print(mse(output_list[1].reconstruction,(output_list[1].q_reconstruction )))
                    #print(mse(output_list[0].reconstruction, eval_models[0].model.decode(output_list[0].z_p,
                    #                        cond_inputs=eval_models[0].model._learned_prior_input(inputs),
                    #                        steps=eval_models[0].model._hp.n_rollout_steps,
                    #                        inputs=inputs)))
                    #print(mse(output_list[1].reconstruction, eval_models[1].model.decode(output_list[1].z_p,
                    #    cond_inputs=eval_models[1].model._learned_prior_input(inputs),
                    #    steps=eval_models[1].model._hp.n_rollout_steps,
                    #    inputs=inputs)))

def get_q_z_space(eval_models):
    for target_model in eval_models:
        target_model.model_test.load_state_dict(target_model.model.state_dict())
        target_model.model_test.eval()
    dataset_class = eval_models[0].conf.data.dataset_spec.dataset_class
    phase = 'val'
    output_list = []
    total_num = len(eval_models)
    for i in range(7):
        data_dir =os.path.join(eval_models[0]._hp.data_dir, 'FL_{}'.format(i))
        val_loader = dataset_class(data_dir, eval_models[0].conf.data, resolution=eval_models[0].model.resolution,
                            phase=phase, shuffle=phase == "train", dataset_size=160). \
            get_data_loader(eval_models[0]._hp.batch_size, 1)
        for target_model in eval_models:
            target_model.evaluator.reset()
        print('Running Testing')
        with autograd.no_grad():            
            for sample_batched in val_loader:
                    inputs = AttrDict(map_dict(lambda x: x.to(eval_models[0].device), sample_batched))
                    for target_model in eval_models:
                        output_list.append(target_model.model_test(inputs))
    for i in range(total_num):

        for j in range(7):
            start = i + (j *total_num)
            if j == 0:
                result = output_list[start].z_p.cpu()
                label = np.full((len(output_list[start].z_p)),j)
            else:
                result = torch.cat([result,output_list[start].z_p.cpu()], dim=0)
                label = np.concatenate((label,np.full((len(output_list[start].z_p)),j)))
        np.savez('/home/workspace/skill/experiments/{}.npz'.format(i),x=result,y=label)




def save_checkpoint(basemodel,folder):
    state = {
            'epoch': 99,
            'global_step':0,
            'state_dict': basemodel.model.state_dict(),
            'optimizer': basemodel.optimizer.state_dict(),
            }
    os.makedirs(folder, exist_ok=True)
    torch.save(state, folder+"/weights_ep100.pth")
    print("Saved checkpoint!")


#def weight_divergence(eval_models):
    '''
    keys = []
    iid_values = []
    non_values = []
    semi_iid_values = []
    for key in non_iid_state_dict.keys():
        if "weight" in key:
            keys.append(key)
            non_values.append(weight_diverse(base_dict[key],non_iid_state_dict[key]))
            iid_values.append(weight_diverse(base_dict[key],iid_state_dict[key]))
            semi_iid_values.append(weight_diverse(base_dict[key],semi_iid_state_dict[key]))
    student_card = pd.DataFrame({'keys':keys,
                             'iid_values':iid_values,
                             'non_values': non_values,
                             "semi_iid_values" : semi_iid_values})
                             
    student_card.to_csv("/home/workspace/skill/experiments/real_result.csv")


    '''



if __name__ == "__main__":

    config = AttrDict(
        path = "spirl/configs/skill_prior_learning/kitchen/FL_clip",
        prefix = "normal-learning with tasks freeze-2",
        new_dir = False,
        dont_save = True,
        resume = "",
        train = True,
        test_prediction = True,
        skip_first_val = False,
        val_sweep = False,
        gpu = 0,
        strict_weight_loading = True,
        deterministic = False,
        log_interval = 500,
        per_epoch_img_logs = 1,
        val_data_size = 160,
        val_interval = 5,
        detect_anomaly = False,
        feed_random_data = False,
        train_loop_pdb = False,
        debug = False,
        save2mp4 = False
    )

    #non_iid_model =  ll_numpy_model_load_change(config=config,prior_path="/home/workspace/skill/experiments/skill_prior_learning/kitchen/prior-task-encoder/weights/round-400-weights.npz",init_path= "/home/workspace/skill/experiments/skill_prior_learning/kitchen/prior-task-encoder/6/round-399-weights.npz")
    #save_path = os.path.join("/home/workspace/skill/experiments/skill_prior_learning/kitchen/prior-task-encoder/6","weights")
    #save_checkpoint(non_iid_model,save_path)
    #basemodel =  pytorch_model_load(config=config, init_path="/home/workspace/skill/experiments/skill_prior_learning/kitchen/FL_hierarchial_cl/normal6/weights/weights_ep99.pth")
    basemodel =  gl_torch_model_load_change(config=config, init_path="/home/workspace/skill/experiments/skill_prior_learning/kitchen/FL_clip/normal-learning with tasks-4/weights/weights_ep95.pth")
    #iid_model =  numpy_model_load(config=config,init_path="/home/workspace/skill/experiments/skill_prior_learning/kitchen/real_iid//weights/round-1000-weights.npz")
    #semi_iid_model =  gl_numpy_model_load(config=config,init_path="/home/workspace/skill/experiments/skill_prior_learning/kitchen/iid_goal_decoder_2/weights/round-500-weights.npz")
    #non_iid_model =  gl_numpy_model_load_change(config=config,init_path="/home/workspace/skill/experiments/skill_prior_learning/kitchen/iid_goal_decoder_2/weights/round-500-weights.npz")
    save_path = os.path.join("/home/workspace/skill/experiments/skill_prior_learning/kitchen/FL_clip/normal-learning with tasks-4","weights")
    save_checkpoint(basemodel,save_path)
    #eval_models = [non_iid_model]
    #get_q_z_space(eval_models)
    #for name, param in basemodel.model.named_parameters():
    #    if name.split('.')[0] == 'classifier' :
    #        pass
    #    else :
    #        param.requires_grad = False
    #basemodel.train()
    '''
    basemodel =  Custome_train2.ModelTrainer(args=config,cid=0,data_dir="./data3/kitchen/",grap_round=0)
    key_value = basemodel.model.state_dict().keys()
    for i in range(50):
        log_lound = (i+1) * 10
        init_path="/home/workspace/skill/experiments/skill_prior_learning/kitchen/FL-iid-4/weights/round-{0}-weights.npz".format(log_lound)
        device = torch.device("cuda")
        np_dict = np.load(init_path,allow_pickle=True)
        params_dict = zip(key_value,np_dict)
        state_dict = OrderedDict()
        for k, v in params_dict:
            state_dict[k] = torch.Tensor(np_dict[v]).to(device)
        np_dict.close()
        l = []
        for d in state_dict :
            if "num_batches_tracked" in d :
                l.append(d)
        for d in l :
            del(state_dict[d])
        basemodel.model.load_state_dict(state_dict,strict = True)
        basemodel.grap_round = log_lound
        basemodel.val()
    

    
    dataset_class = basemodel.conf.data.dataset_spec.dataset_class
    
    phase = 'val'
    data_dir = "./data7/kitchen/FL_0/"
    val_loader = dataset_class(data_dir, basemodel.conf.data, resolution=basemodel.model.resolution,
                            phase="train", shuffle=phase == "train", dataset_size=160). \
            get_data_loader(basemodel._hp.batch_size, 1)
    total_score = []
    for i in range(100):
        #data_dir =os.path.join(semi_iid_model._hp.data_dir, 'FL_{}'.format(i))
        print('Running Testing')
        with autograd.no_grad():
            for sample_batched in val_loader:
                    inputs = AttrDict(map_dict(lambda x: x.to(basemodel.device), sample_batched))
                    # run non-val-mode model (inference) to check overfitting
                    output = basemodel.model_test(inputs)
                    argmax_tensor = torch.argmax(output.skill, dim=1)
                    indices_equal = torch.eq(inputs.tasks, argmax_tensor)
                    score = indices_equal.sum().item() / indices_equal.numel()
                    total_score.append(score)
                    print(score)
    print(np.mean(np.array(total_score)))
    
                    if result1 == []:
                        result1 = output1.z
                    else:
                        result1 = torch.cat([result1,output1.z], dim=0)
                    if label1 == []:
                        label1 = np.full((init_model1._hp.batch_size),i)
                    else:
                        label1 = np.concatenate((label1,np.full((init_model1._hp.batch_size),i)))
                    if result2 == []:
                        result2 = output1.z
                    else:
                        result2 = torch.cat([result2,output1.z], dim=0)
                    if label2 == []:
                        label2 = np.full((init_model1._hp.batch_size),i)
                    else:
                        label2 = np.concatenate((label2,np.full((init_model1._hp.batch_size),i)))
                    break
    #np.savez('/home/workspace/skill/experiments/normal_z.npz',x=result2,y=label2)
    #np.savez('/home/workspace/skill/experiments/FL_z.npz',x=result1,y=label1)
    '''