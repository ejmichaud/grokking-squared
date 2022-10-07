import numpy as np
import torch
import torch.nn as nn
import copy


#--------------------------Neural Network---------------------------#
class NET(nn.Module):
    # 2 hidden layer MLP
    def __init__(self, input_dim, output_dim, w):
        super(NET, self).__init__()
        self.l1 = nn.Linear(input_dim, w)
        self.l2 = nn.Linear(w, w)
        self.l3 = nn.Linear(w, output_dim)

    def forward(self, x):
        self.x1 = torch.tanh(self.l1(x))
        self.x2 = torch.tanh(self.l2(self.x1))
        self.x3 = self.l3(self.x2)
        return self.x3

class DEC(nn.Module):
    def __init__(self, reprs_dim, output_dim, w):
        super(DEC, self).__init__()
        self.net = NET(reprs_dim, output_dim, w)

    def forward(self, reprs, x_id):
        self.add1 = reprs[x_id[:,0]]
        self.add2 = reprs[x_id[:,1]]
        # hard code addition
        self.add = self.add1 + self.add2
        self.out = self.net(self.add)
        return self.out


# addition toy
def train_add(p=10,
          reprs_dim=1,
          output_dim=30,
          train_num=45,
          seed=58,
          steps=5000,
          eff_steps = 5000,
          batch_size=45,
          init_scale_reprs=1.0,
          init_scale_nn=1.0,
          label_scale=1.0,
          eta_reprs=1e-3,
          eta_dec=1e-4,
          log_freq=1000,
          width=200,
          weight_decay_reprs=0.0,
          weight_decay_dec=0.0,
          threshold_train_acc=0.9,
          threshold_test_acc=0.9,
          threshold_rqi=0.95,
          threshold_P=0.01,
          loss_type="MSE"):

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if batch_size > train_num:
        batch_size = train_num
        print("batch size larger than the training set. We have set batch size=training size.")
        
    if loss_type == "CE":
        output_dim = 2*p - 1
        print("Using cross entropy, setting output_dim=2p-1={}".format(output_dim))

    device = torch.device("cpu")#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    
    reprss = []
    reprss_scale = []


    y_templates = np.random.normal(0,1,size=(2*p-1, output_dim))*label_scale
    y_templates = torch.tensor(y_templates, dtype=torch.float, requires_grad=True).to(device)
    
    
    #-----------------------------Dataset------------------------------#
    # the full dataset contains p(p+1)/2 samples. Each sample has input (a, b).
    all_num = p*(p+1)//2 # for addition (abelian group), we deem a+b and b+a the same sample
    D0_id = []
    xx_id = []
    yy_id = []
    for i in range(p):
        for j in range(i,p):
            D0_id.append((i,j))
            xx_id.append(i)
            yy_id.append(j)
    xx_id = np.array(xx_id)
    yy_id = np.array(yy_id)

    # P0 includes all parallelograms (a,b,c,d) in the full dataset
    # A parallelogram means (a,b), (c,d) are training samples and a+b=c+d
    P0 = []
    P0_id = []
    ii = 0
    for i in range(all_num):
        for j in range(i+1,all_num):
            if np.sum(D0_id[i]) == np.sum(D0_id[j]):
                P0.append(frozenset({D0_id[i], D0_id[j]}))
                P0_id.append(ii)
                ii += 1
    P0_num = len(P0_id)

    # A is equivalent to P0, but converts parallelograms (geometry) 
    # to coefficients of linear equations (algebra), ready for further analysis.
    # For example, p=4, P0 contains a parallelogram (0,3,1,2)
    # This translates to a row in A [1, -1, -1, 1], meaning E0 - E1 - E2 + E3 = 0
    A = []
    eq_id = 0
    for i1 in range(P0_num):
        i,j = list(P0[i1])[0]
        m,n = list(P0[i1])[1]
        if i+j==m+n:
            x = np.zeros(p,)
            x[i] = x[i] + 1
            x[j] = x[j] + 1
            x[m] = x[m] - 1
            x[n] = x[n] - 1
            A.append(x)
            eq_id = eq_id + 1
    A = np.array(A).astype(int)
    
    # draw a subset from the full set as training set D
    train_id = np.random.choice(all_num,train_num, replace=False)
    test_id = np.array(list(set(np.arange(all_num)) - set(train_id)))
    inputs_id = np.transpose(np.array([xx_id,yy_id]))
    out_id = (xx_id + yy_id)

    print("----------------------------------------")
    print("Task 1: Analyzing the dataset before training...")
    # P0(D) is P_0(D) in paper: it includes all the parallelograms in training set D
    P0D_id = []
    ii = 0
    for i in range(all_num):
        for j in range(i+1,all_num):
            if np.sum(D0_id[i]) == np.sum(D0_id[j]):
                if i in train_id and j in train_id:
                    P0D_id.append(ii)
                ii += 1
    P0D = []
    for i in P0D_id:
        P0D.append(P0[i])
    # P0D_c includes the parallelograms not in P0(D), but in P0. 'c' means complement.
    P0D_c_id = set(P0_id) - set(P0D_id)


    # PD is P(D) in paper. PD includes P0D and all parallelograms induced from P0D.
    # How does induction work?
    # Example: (0,3,1,2) being a parallelogram and (2,5,3,4) being a parallelogram
    # induce that (0,5,1,4) is also a parallelogram.
    # This geometric argument is translated to linear dependence algebraically,
    # i.e., {E0+E3=E1+E2, E2+E5=E3+E4} -> {E0+E5=E1+E4}.
    PD_id = []
    mat = A[P0D_id]
    eigs = np.linalg.eigh(np.matmul(np.transpose(mat),mat))[0]
    null_dim = np.sum(eigs < 1e-8)

    # a parallelogram can be induced from P0(D) if it is linearly dependent on P0(D).
    # linear dependence <=> the rank (of mat) does not change after adding the parallelogram.
    # linear independence <=> the rank (of mat) increases by one after adding the parallelogram.
    for i in P0D_c_id:
        P0D_id_aug = copy.deepcopy(P0D_id)
        P0D_id_aug.append(i)
        mat_aug = A[P0D_id_aug]
        P0D_aug = []
        for j in P0D_id_aug:
            P0D_aug.append(P0[j])
        null_dim_aug = np.sum(np.linalg.eigh(np.matmul(np.transpose(mat_aug),mat_aug))[0] < 1e-8)
        if null_dim_aug == null_dim:
            PD_id.append(i)

    PD_id = PD_id + P0D_id

    PD = []
    for i in PD_id:
        PD.append(P0[i])
        
    # Dbar(D) contains all the examples that can be got correctly (ideally) given training data
    # One may ask: given a test sample i+j, how can the neural network know its answer if never seen it?
    # The answer is via a good representation, i.e., parallelograms.
    # If there exists a training sample m+n such that i+j=m+n, and (i,j,m,n) is a parallelogram (i.e., Ei+Ej=Em+En)
    # then Dec(Ei+Ej) = Dec(Em+En) = Y_{m+n} = Y_{i+j}, i.e., the nueral network can get i+j correct.
    Dbar_id = list(copy.deepcopy(train_id))

    for i1 in test_id:
        flag = 0
        for j1 in train_id:
            i, j = D0_id[i1]
            m, n = D0_id[j1]
            if {(i,j),(m,n)} in PD:
                flag = 1
                break
        if flag == 1:
            Dbar_id.append(i1)

    # Given training data, without training,
    # we are able to determine the ideal accuracy, denoted as \overline{\rm Acc} in the paper.
    # Empirical accuracy (by training nn, denoted as {\rm Acc}),
    # is upper bounded by the ideal accuracy (except for luck).
    acc_ideal = len(Dbar_id)/all_num # the full dataset
    acc_ideal_test = (len(Dbar_id)-len(train_id))/len(test_id) # test set
    print("acc_ideal_test = {}/{}={}".format(len(Dbar_id)-len(train_id),len(test_id),acc_ideal_test))
    print("the degree of freedom (except translation/scaling) for the reprsentation is {}".format(null_dim-2))
    print("dof=0 means the linear repr is the unique repr, while dof>0 means existence of other reprs")

    # embedding input and output digits to random vectors at initialization, which are trainable.
    reprs = torch.nn.parameter.Parameter((torch.rand(p,reprs_dim)-1/2).to(device)*init_scale_reprs)
    
    labels_train = y_templates[out_id[train_id]].detach().clone().requires_grad_(True)
    in_id_train = inputs_id[train_id]
    out_id_train = out_id[train_id]
    
    labels_test = y_templates[out_id[test_id]].detach().clone().requires_grad_(True)
    in_id_test = inputs_id[test_id]
    out_id_test = out_id[test_id]


    # initialize the decoder. init_scale_nn is the initialization scale.
    model = DEC(reprs_dim, output_dim, width).to(device)
    for p_ in model.net.parameters():
        p_.data = p_.data * init_scale_nn

    # collect statistics
    losses_train = []
    losses_test = []
    accs_train = []
    accs_test = []
    rqis = []
    reprss = []
    reprss_scale = []

    # use different optimizers for representations and the decoder.
    optimizer1 = torch.optim.AdamW({reprs}, lr=eta_reprs, weight_decay = weight_decay_reprs)
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=eta_dec, weight_decay = weight_decay_dec, betas=(0.9,0.999))

    # indicate whether metrics (training/test accuracy, RQI) rise above certain thresholds
    reach_acc_train = False
    reach_acc_test = False
    reach_rqi = False
    
    print("----------------------------------------")
    print("Task 2: Training with neural network...")

    # make some parallelograms
    parallelograms = []
    for i in range(p):
        for j in range(i+1,p):
            for m in range(j,p):
                for n in range(m+1,p):
                    if (i+n-j-m) == 0:
                        parallelograms.append([i,n,j,m])
    parallelograms = torch.tensor(parallelograms).to(device)
    num_P_ideal = len(parallelograms)



    for step in range(steps):  # loop over the dataset multiple times

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # random batch
        choice = np.random.choice(np.arange(train_num), batch_size, replace=False)
        
        # caluate loss for train/test
        outputs_train = model(reprs, in_id_train[choice])
        outputs_test = model(reprs, in_id_test)
        if loss_type == "MSE":
            loss_train = torch.mean((outputs_train-labels_train[choice])**2)
            loss_test = torch.mean((outputs_test-labels_test)**2)
        else:
            loss_train = nn.CrossEntropyLoss()(outputs_train, torch.tensor(out_id_train[choice], dtype=torch.long))
            loss_test = nn.CrossEntropyLoss()(outputs_test, torch.tensor(out_id_test, dtype=torch.long))
            
        losses_train.append(loss_train.item())
        losses_test.append(loss_test.item())
        
        # update weights
        loss_train.backward()
        optimizer1.step()
        optimizer2.step()

        # calculate accuracy for train/test
        outputs_train = model(reprs, in_id_train)
        if loss_type == "MSE":
            pred_train_id = torch.argmin(torch.sum((outputs_train.unsqueeze(dim=1) - y_templates.unsqueeze(dim=0))**2, dim=2), dim=1)
            pred_test_id = torch.argmin(torch.sum((outputs_test.unsqueeze(dim=1) - y_templates.unsqueeze(dim=0))**2, dim=2), dim=1)
        else:
            pred_train_id = torch.argmax(outputs_train, dim=1)
            pred_test_id = torch.argmax(outputs_test, dim=1)
        acc_nn_train = np.mean(pred_train_id.cpu().detach().numpy() == out_id_train)
        accs_train.append(acc_nn_train)
        
        outputs_test = model(reprs, in_id_test)
        acc_nn_test = np.mean(pred_test_id.cpu().detach().numpy() == out_id_test)
        accs_test.append(acc_nn_test)

        if not reach_acc_train:
            if acc_nn_train >= threshold_train_acc:
                reach_acc_train = True
                iter_train = step
                
        if not reach_acc_test:
            if acc_nn_test >= threshold_test_acc:
                reach_acc_test = True
                iter_test = step

        if step % log_freq == 0:
            print("step: %d  | loss: %.8f "%(step, loss_train.cpu().detach().numpy()))

        # normalized representations (zero mean, unit variance) 
        reprs_scale = (reprs-reprs.mean(0).unsqueeze(dim=0))/reprs.std(0).unsqueeze(dim=0)
        
        #num_P_ideal: the number of all possible parallelogram
        #num_P_real: the number of parallelogram actually appearing in the representation after training
        dists = reprs_scale[parallelograms[:,:2]].sum(axis=1) - reprs_scale[parallelograms[:,2:]].sum(axis=1)
        num_P_real = torch.sum((dists**2).mean(1) < threshold_P).item()
                            
        # define RQI as ratio of the number of real vs ideal (all) parallelograms
        rqi = num_P_real/num_P_ideal
        if not reach_rqi:
            if rqi > threshold_rqi:
                reach_rqi = True
                iter_rqi = step
                
        # Given training set D and representation R after training, out theory can predict the test accuracy
        PR = []
        PR_id = []
        for ii in range(P0_num):
            i, j = list(P0[ii])[0]
            m, n = list(P0[ii])[1]
            dist = reprs_scale[i] + reprs_scale[j] - reprs_scale[m] - reprs_scale[n]
            if (torch.mean(dist**2)<threshold_P):
                PR_id.append(ii)
                PR.append(P0[ii])
                
        # Dbar(D,P). Note this is different from Dbar(D).
        Dbar_P_id = list(copy.deepcopy(train_id))

        for i1 in test_id:
            flag = 0
            for j1 in train_id:
                i, j = D0_id[i1]
                m, n = D0_id[j1]
                if {(i,j),(m,n)} in PR:
                    flag = 1
                    break
            if flag == 1:
                Dbar_P_id.append(i1)
                
        acc_pred_test = (len(Dbar_P_id)-len(train_id))/len(test_id)
        

        rqis.append(rqi)
        reprss.append(copy.deepcopy(reprs.cpu().detach().numpy()))
        reprss_scale.append(copy.deepcopy(reprs_scale.cpu().detach().numpy()))


    if not reach_acc_train:
        iter_train = step
        
    if not reach_acc_test:
        iter_test = step

    if not reach_rqi:
        iter_rqi = step

    rqis = np.array(rqis)
    losses_train = np.array(losses_train)
    losses_test = np.array(losses_test)
    accs_train = np.array(accs_train)
    accs_test = np.array(accs_test)
    reprss = np.array(reprss)
    reprss_scale = np.array(reprss_scale)
    
    print("final train acc=%.4f, test acc=%.4f, RQI=%.4f"%(acc_nn_train, acc_nn_test, rqi))
    print("Steps to reach thresholds: train acc={}, test acc={}, RQI={}".format(iter_train, iter_test, iter_rqi))
    
    
    #---------------------effective theory-------------------#
    print("----------------------------------------")
    print("Task 3: Training with effective loss...")
    E = reprss_scale[0,:,0]
    Z0 = np.sum(E**2)
    l0 = np.sum(np.sum(A*E[np.newaxis,:], axis=1)**2)
    temp = np.sum(np.sum(A*E[np.newaxis,:], axis=1)[:,np.newaxis]*A, axis=0)
    dE = 2*l0/Z0**2*E - 2/Z0*temp
    Es_eff = []
    losses_eff = []

    step = eta_reprs
    n_step = eff_steps

    for i in range(n_step):
        Es_eff.append(copy.deepcopy(E))
        l0 = np.sum(np.sum(A*E[np.newaxis,:], axis=1)**2)
        losses_eff.append(l0)
        temp = np.sum(np.sum(A*E[np.newaxis,:], axis=1)[:,np.newaxis]*A, axis=0)
        dE = 2*l0/Z0**2*E - 2/Z0*temp
        E = E + step*dE
        if i % log_freq == 0:
            print("step: %d  | loss: %.8f "%(i, l0))
    Es_eff = np.array(Es_eff)    
    losses_eff = np.array(losses_eff)
    print("saving trajectories...")
    
    # collect return results in a dictionary
    # task 1 data
    dic = {}
    dic["all_num"] = all_num
    dic["train_num"] = train_num
    dic["test_num"] = all_num - train_num
    dic["train_ratio"] = train_num/all_num
    dic["test_ratio"] = 1 - train_num/all_num
    dic["ideal_test_acc"] = acc_ideal_test
    dic["ideal_acc"] = acc_ideal
    dic["pred_test_acc"] = acc_pred_test
    dic["pred_acc"] = ((all_num - train_num)*acc_pred_test + train_num*1.0)/all_num
    dic["dof"] = null_dim
    
    # task 2 data
    dic["loss_train"] = losses_train
    dic["loss_test"] = losses_test
    dic["acc_train"] = accs_train
    dic["acc_test"] = accs_test
    dic["acc"] = ((all_num - train_num)*accs_test + train_num*accs_train)/all_num
    dic["repr_nn"] = reprss
    dic["repr_normalized_nn"] = reprss_scale
    dic["rqi"] = rqis
    dic["iter_train"] = iter_train
    dic["iter_test"] = iter_test
    dic["iter_rqi"] = iter_rqi
    
    # task 3 data
    dic["repr_eff"] = Es_eff
    dic["loss_eff"] = losses_eff
    
    # configuration parameters
    dic["p"] = p
    dic["repr_dim"] = reprs_dim
    dic["seed"] = seed
    dic["steps"] = steps
    dic["eff_steps"] = eff_steps
    dic["batch_size"] = batch_size
    dic["init_scale_reprs"] = init_scale_reprs
    dic["init_scale_nn"] = init_scale_nn
    dic["label_scale"] = label_scale
    dic["eta_repr"] = eta_reprs
    dic["eta_dec"] = eta_dec
    dic["width"] = width
    dic["weight_decay_repr"] = weight_decay_reprs
    dic["weight_decay_dec"] = weight_decay_dec
    dic["threshold_train_acc"] = threshold_train_acc
    dic["threshold_test_acc"] = threshold_test_acc
    dic["threshold_rqi"] = threshold_rqi
    dic["threshold_P"] = threshold_P
    dic["loss_type"] = loss_type
    return dic
    

#train_add( reprs_dim = 2)
