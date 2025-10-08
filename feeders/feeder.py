import sys
sys.path.extend(['../'])

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from feeders import tools
import os 

class Feeder(Dataset):
    def __init__(self, data_path, label_path, length_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, data_degradation=False,
                 data_degradation_type=None, drop_rate=0.0, structured_degredation=False, 
                 structured_degredation_type=None, structured_res=1, FPS=30, chunks=1,
                 mitigation=False):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.length_path = length_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.data_degradation = data_degradation
        self.data_degradation_type = data_degradation_type
        self.drop_rate = drop_rate
        self.structured_degredation = structured_degredation
        self.structured_degredation_type = structured_degredation_type
        self.structured_res = structured_res
        self.FPS = FPS
        self.chunks = chunks
        self.mitigation = mitigation

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        try: 
            self.length = np.load(self.length_path)
        except:
            self.length = np.zeros((self.label.shape[0],))

        np.save('train_length.npy', self.length)

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.length = self.length[0:100]
            self.sample_name = self.sample_name[0:100]
        
        self.raw_data = self.data.copy()

        if self.data_degradation and self.structured_degredation:
            sys.exit('Cannot have both data_degradation and structured_degradation set to True')

        if self.data_degradation:
            if self.data_degradation_type is None or self.data_degradation_type not in ['ucl', 'delete', 'repeat_previous', 'repeat_next', 'interpolate']:
                raise ValueError('data_degradation_type must be one of ["ucl", "delete", "repeat_previous", "repeat_next", "interpolate"]')
            else:
                print('Data degradation type: ', self.data_degradation_type, 'with drop rate: ', self.drop_rate)
            
            if self.data_degradation_type == 'ucl':
                N, C, T, V, M = self.data.shape #(X, 3, 300, 25, 2) (N,C,T,V,M)
                data = np.zeros((N, C, T, V, M))
                for n in tqdm(range(N)):
                    #if n==0:
                    #    np.savetxt('before2.txt', self.data[n,0,:,:,0])
                    drop_index = np.random.choice(range(1, int(self.length[n].item())), size=int(
                        self.drop_rate * int(self.length[n].item())), replace=False)
                    if self.length[n].item() <= 1:
                        continue
                    tmp_deleted = np.delete(self.data[n], drop_index, 1)[
                        :, :int(self.length[n].item() * (1 - self.drop_rate))]
                    
                    data[n] = self.get_refill(tmp_deleted, T, n)

                self.data = data

            elif self.data_degradation_type == 'delete':
                N, C, T, V, M = self.data.shape #(X, 3, 300, 25, 2) (N,C,T,V,M)
                data = np.zeros((N, C, T, V, M))
                for n in tqdm(range(N)):
                    #if n==0:
                    #    np.savetxt('before.txt', self.data[n,0,:,:,0])
                    drop_indices = np.random.choice(range(1, int(self.length[n].item())), 
                                                    size=int((self.drop_rate) * int(self.length[n].item())), replace=False)
                    drop_indices = np.sort(drop_indices) # sort the drop index so we don't include data we intend to delete in the next step.
                    if self.length[n] <= 1:
                        continue
                    
                    trans_data = self.data[n,:,:int(self.length[n].item()),:,:]
                    new_data = np.delete(trans_data, drop_indices, axis=1)
                    data[n,:,:new_data.shape[1],:,:] = new_data
                    self.length[n] = new_data.shape[1]
                
                    data[n] = self.get_refill(data[n,:,:int(self.length[n])], T, n)

                self.data = data

            elif self.data_degradation_type == 'repeat_next':
                N, C, T, V, M = self.data.shape
                data = self.data.copy()
                for n in tqdm(range(N)):
                    #if n==0:
                    #    np.savetxt('before_repeat_next.txt', self.data[n,0,:,:,0])
                    stream_length = self.length[n].item()
                    if stream_length <= 1:
                        continue
                    drop_indices = np.random.choice(range(1, int(stream_length)), 
                                                size=int(self.drop_rate * int(stream_length)), replace=False)
                    drop_indices = np.sort(drop_indices)
                    rev_drop_indices = drop_indices[::-1]

                    for j in range(len(rev_drop_indices)):
            
                        if rev_drop_indices[j] != T-1:
                            data[n, :, rev_drop_indices[j]] = data[n, :, rev_drop_indices[j]+1]
                    
                    data[n] = self.get_refill(data[n,:,:int(self.length[n])], T, n)
                    #if n==32:
                    #    np.savetxt('after_repeat_next_'+ str(self.drop_rate) + '.txt', data[0:32,0,:,0,0])
                    #    sys.exit()

                self.data = data

            elif self.data_degradation_type == 'repeat_previous':
                N, C, T, V, M = self.data.shape
                data = self.data.copy()
                for n in tqdm(range(N)):
                    #if n==0:
                    #    np.savetxt('before_interpolate.txt', self.data[n,0,:,:,0])
                    stream_length = self.length[n].item()
                    if stream_length <= 1:
                        continue
                    drop_indices = np.random.choice(range(1, int(stream_length)), 
                                                size=int(self.drop_rate * int(stream_length)), replace=False)
                    drop_indices = np.sort(drop_indices) # sort the drop index so we don't include data we intend to delete in the next step.

                    for j in range(len(drop_indices)):
                        if drop_indices[j] != 1:
                            data[n, :, drop_indices[j]] = data[n, :, drop_indices[j]-1] 
                        else:
                            continue
                    
                    data[n] = self.get_refill(data[n,:,:int(self.length[n])], T, n)
                    #if n==32:
                        #np.savetxt('after_repeat_previous_'+ str(self.drop_rate) + '.txt', data[n,0,:,:,0]) # 1 samples, all 25 joints, 1 coord, 1 person, all frames
                    #    np.savetxt('after_repeat_previous_'+ str(self.drop_rate) + '.txt', data[0:32,0,:,0,0]) # 32 samples, 1 joint, 1 coord, 1 person, all frames
                    #    sys.exit()
                    
                self.data = data

            elif self.data_degradation_type == 'interpolate':
                N, C, T, V, M = self.data.shape #(X, 3, 300, 25, 2) (N,C,T,V,M)
                data = self.data.copy()
                
                for n in tqdm(range(N)):
                    #if n==0:
                        #np.savetxt('before_interpolate.txt', self.data[n,0,:,:,0])
                        #np.savetxt('before_interpolate.txt', self.data[0:32,0,:,0,0])
                    
                    if self.length[n].item() <= 1:
                        continue
                    drop_indices = np.random.choice(range(1, int(self.length[n].item())), 
                                                size=int(self.drop_rate * int(self.length[n].item())), replace=False)
                    #if n == 0:
                    #    print(self.drop_rate)
                    #    print(self.length[n].item())
                    #    print(range(1, int(self.length[n].item())))
                    #    print(int(self.drop_rate * int(self.length[n].item())))


                    drop_indices = np.sort(drop_indices)
                    sub_arrays = np.split(drop_indices, np.flatnonzero(np.diff(drop_indices)!=1) + 1)
                    for sub_array in sub_arrays:
                        len_sub_array = len(sub_array)
                        if len_sub_array > 1:
                            for j in range(len_sub_array): 
                                if sub_array[len_sub_array-1]+1 == T:
                                    end = self.data[n,:,sub_array[len_sub_array-1],:,:]
                                else:
                                    end = self.data[n,:,sub_array[len_sub_array-1]+1,:,:]
                                start = self.data[n,:,sub_array[0]-1,:,:]
                                
                                inc = (end-start) / (len_sub_array+1)
                                data[n,:,sub_array[j],:,:] = inc * (j+1) + start
                        else:
                            if len(sub_array) == 0:
                                continue

                            if sub_array[0] == 299:
                                data[n,:,sub_array[0],:,:] = self.data[n,:,sub_array[0],:,:]
                            else:
                                data[n,:,sub_array] = (self.data[n,:,sub_array[0]-1] + self.data[n,:,sub_array[0]+1]) / 2

                    data[n] = self.get_refill(data[n,:,:int(self.length[n])], T, n)
                    #if n==32:
                        #np.savetxt('after_interpolate_'+ str(self.drop_rate) + '.txt', data[n,0,:,:,0])
                    #    np.savetxt('after_interpolate_' + str(self.drop_rate) + '.txt', data[0:32,0,:,0,0])
                    #    sys.exit()
                    
                self.data = data

            ## saving degraded_data_routine 
            # save degraded data in batches of 32

            #total_samples = N
            #bs = 32
            #number_of_batches = total_samples // bs + 1 # for w/e is left over
            #location = 'degraded_dataset/' + str(self.data_degradation_type) + '/dr_' + str(self.drop_rate) + '/' 
            #if not os.path.exists(location):
            #    os.makedirs(location)
            #for i in tqdm(range(number_of_batches)):
            #    np.save(location + str(i) + '.npy', self.data[i*bs:(i+1)*bs])
            #        np.save('degraded_data_' + str(self.drop_rate) + '_' + str(i) + '.npy', self.length[i*bs:(i+1)*bs])
            #sys.exit()
        elif self.structured_degredation is True:
            if self.structured_degredation_type is None or self.structured_degredation_type not in ['reduced_resolution', 'frame_rate']:
                raise ValueError('structured_degradation_type must be one of ["reduced_resolution", "frame_rate"]')
            elif self.structured_degredation_type == 'reduced_resolution':
                print('Data degradation type: ', self.structured_degredation_type, 'with frequence: ', self.structured_res)
            else:
                print('Data degradation type: ', self.structured_degredation_type, 'with FPS: ', self.FPS)
            
            if self.structured_degredation_type == 'reduced_resolution':

                N, C, T, V, M = self.data.shape
                data = np.zeros((N, C, T, V, M))
                for n in tqdm(range(N)):
                    #if n==0:
                    #    np.savetxt('./testing/reduced_res_before_' + str(self.structured_res) +'.txt', self.data[n,0,:,0,0])
                    
                    if self.mitigation is True:

                        no_of_frames = self.length[n]
                        data[n,:,:int(no_of_frames):self.structured_res,:,:] = self.data[n, :, :int(no_of_frames):self.structured_res, :, :]
                        
                        x = np.arange(no_of_frames)
                        y = x[:int(no_of_frames):self.structured_res]

                        selected_frames = np.where(np.isin(x,y))[0]
                        all_indices = np.arange(len(x))
                        unselected_indices = np.setdiff1d(all_indices, selected_frames)

                        sub_arrays = np.split(unselected_indices, np.flatnonzero(np.diff(unselected_indices.T)!=1) + 1)
                        for sub_array in sub_arrays:
                            if len(x)-1 in sub_array:
                                continue
                            len_sub_array = len(sub_array)
                            if len_sub_array > 1:
                                for i in range(len_sub_array):
                                    
                                    if sub_array[len_sub_array-1]+1 == len(x):
                                        end = self.data[n,:,sub_array[len_sub_array-1],:,:]
                                    else:
                                        end = self.data[n,:,sub_array[len_sub_array-1]+1,:,:]
                                    start = self.data[n,:,sub_array[0]-1,:,:]
                                    
                                    inc = (end-start) / (len_sub_array+1)
                                    data[n,:,sub_array[i],:,:] = inc * (i+1) + start
                            else:
                                if len(sub_array) == 0:
                                    continue
                                data[n,:,sub_array[0]] = (self.data[n,:,sub_array[0]-1,:,:] + self.data[n,:,sub_array[0]+1,:,:]) / 2
                    else:
                        no_of_frames = self.length[n]
                        tmp = self.data[n, :, :int(no_of_frames):self.structured_res, :, :]
                        data[n,:,:tmp.shape[1],:,:] = tmp
                        #if n==0:
                        #    print(data[n].shape, tmp.shape, int(no_of_frames), self.structured_res)
                        self.length[n] = int(tmp.shape[1])

                    data[n] = self.get_structured_refill(data[n,:,:int(self.length[n])], T, n)
                    #if n==0:
                    #    np.savetxt('./testing/reduced_res_after_' + str(self.structured_res) +'.txt', data[n,0,:,0,0])

                self.data = data
            elif self.structured_degredation_type == 'frame_rate':

                if self.FPS == 30:
                    sys.exit('Cannot have the same frame rate as original data')

                N, C, T, V, M = self.data.shape
                data = np.zeros((N, C, T, V, M))
                FPS_Drop = self.FPS/30

                for n in tqdm(range(N)):
                    chunk_length = int(self.length[n].item() * FPS_Drop)
                    if chunk_length < 1:
                        continue
                    max_chunk_start = int(self.length[n].item() - chunk_length)
                    chunk_start = np.random.randint(0, high=max_chunk_start) # randint is exclusive of high

                    if self.mitigation is True:
                        if n==0:
                            print('Mitigation applied')
                        data[n] = self.data[n]

                        start = self.data[n, :,chunk_start, :, :]
                        end = self.data[n, :,chunk_start+chunk_length, :, :]

                        for i in range(0, chunk_length):
                            increment = (end-start) / chunk_length
                            data[n,:,chunk_start+i,:,:] = start + increment * i
                    else:
                        delete_indices = np.arange(chunk_start, chunk_start+chunk_length)
                        #if n==0:
                        #    print(self.length[n], chunk_length, delete_indices.shape)#, delete_indices)
                        arr = np.delete(self.data[n].copy(), delete_indices, axis=1)
                        data[n, :, :arr.shape[1], :, :] = arr
                        self.length[n] = self.length[n] - chunk_length
                        


                    arr = self.data[n, :, chunk_start:chunk_start+chunk_length, :, :]
                    data[n,:,:arr.shape[1],:,:] = arr
                    #print(n, self.length[n].item(), FPS_Drop, chunk_length, max_chunk_start, chunk_start, arr.shape)

                    self.length[n] = int(arr.shape[1])
                    data[n] = self.get_structured_refill(data[n,:,:int(self.length[n])], T, n)
                
                self.data = data




    def get_refill(self, degraded_data, T, n):

        rest = T - degraded_data.shape[1]
        num = int(np.ceil(rest / degraded_data.shape[1]))
        pad = np.concatenate([degraded_data
                            for _ in range(num + 1)], 1)[:, :T]
        #if n==0:
        #    np.savetxt('after2.txt', data[n,0,:,:,0])
        self.length[n] = int(self.length[n].item() * (1 - self.drop_rate))

        return pad
            
    def get_structured_refill(self, degraded_data, T, n):

        rest = T - degraded_data.shape[1]
        num = int(np.ceil(rest / degraded_data.shape[1]))
        pad = np.concatenate([degraded_data
                            for _ in range(num + 1)], 1)[:, :T]
        
        return pad
        
    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, self.length[index], index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)


if __name__ == '__main__':
    import os
    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
