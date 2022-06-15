import gzip
import numpy as np

def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))

def main():
    data = 0
    last_n = 75
    with gzip.open('mnistGNN/mnist_to_graph/train-images-idx3-ubyte.gz', 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape([-1,28,28])
    
    tmp = np.empty([60000, 28, 28], dtype=np.int64)
    iter = 0
    for i in data :
        
        sort_data = k_largest_index_argsort(i, last_n)
        smallest_image_node = i[sort_data[-1][0], sort_data[-1][1]]
        tmp[iter] = data[iter]
        a = tmp[iter].reshape(784)
        cnt = 1
        for j in range(len(a)):
            if a[j] >= smallest_image_node and cnt<= last_n:
                cnt += 1
                a[j] = 1000
                # tmp[iter] = np.where(tmp[iter] < smallest_image_node, -1, 1000)
            else:
                a[j] = -1

        tmp[iter] = a.reshape(28,28)
        iter += 1
        print(iter)


    for e,imgtmp in enumerate(tmp):
        img = np.pad(imgtmp,[(2,2),(2,2)],"constant",constant_values=(-1))
        cnt = 0

        for i in range(2,30):
            for j in range(2,30):
                if img[i][j] == 1000:
                    img[i][j] = cnt
                    cnt+=1
        
        edges = []
        # y座標、x座標
        npzahyou = np.zeros((cnt,2))

        for i in range(2,30):
            for j in range(2,30):
                if img[i][j] == -1:
                    continue

                #8近傍に該当する部分を抜き取る。
                filter = img[i-2:i+3,j-2:j+3].flatten()
                filter1 = filter[[6,7,8,11,13,16,17,18]]

                npzahyou[filter[12]][0] = i-2
                npzahyou[filter[12]][1] = j-2

                for tmp in filter1:
                    if not tmp == -1:
                        edges.append([filter[12],tmp])
        print(e)
        np.save("mnistGNN/dataset/graphs/"+str(e),edges)
        np.save("mnistGNN/dataset/node_features/"+str(e),npzahyou)

if __name__=="__main__":
    main()