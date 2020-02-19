import numpy as np
from tqdm import tqdm
from cvxopt import matrix, solvers
from sklearn.cluster import KMeans
from collections import Counter
import sys

def perform_kmeans(datapoints, clusters):
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(datapoints)

    theta = np.zeros((len(datapoints), clusters))
    for ii in range(len(datapoints)):
        theta[ii][kmeans.labels_[ii]] = 1.0

    return theta

def update_alphas(data, theta, nu):
    N = len(data)
    C = 1.0 / (N * nu)

    P = np.zeros((N, N))
    q = np.zeros(N)
    for ii in range(N):
        for jj in range(ii, N):
            val = 2.0 * np.dot(data[ii], data[jj]) * np.dot(theta[ii], theta[jj])
            # val = 2.0 * np.dot(data[ii], data[jj]) * 1.0
            P[ii][jj] = val
            P[jj][ii] = val
        q[ii] = -np.dot(data[ii], data[ii])
    G = np.vstack((-np.eye(N), np.eye(N)))
    h = np.concatenate((np.zeros(N), C * np.ones(N)))
    A, b = np.transpose(theta), np.ones(len(theta[0]))

    # print C
    # print P.shape, q.shape
    # print G.shape, h.shape
    # print A.shape, b.shape
    P, q = matrix(P, tc = 'd'), matrix(q, tc = 'd')
    G, h = matrix(G, tc = 'd'), matrix(h, tc = 'd')
    A, b = matrix(A, tc = 'd'), matrix(b, tc = 'd')

    sol = solvers.qp(P,q,G,h,A,b)

    print('Optimization status:', sol['status'])
    print('Primal objective:', sol['primal objective'])
    alphas = np.array(sol['x'])

    tmp = alphas.flatten()
    thres = 1e-6
    cnt = 0
    for tt in tmp:
        if tt > thres:
            cnt += 1
    # print 'nu: ', nu
    # print 'SVDD Boundry Error:', cnt
    # print 'N:', len(tmp), 'P:', (100.0 * cnt) / len(tmp)
    return alphas.flatten(), sol['status']

def compute_centers_and_radius(data, alphas, thetas, nu):
    N = len(data)
    num_centers = thetas.shape[1]
    centers = []
    radii = []
    for kk in range(num_centers):
        tmp = np.zeros(data[0].shape)
        for ii in range(N):
            tmp += data[ii] * alphas[ii] * thetas[ii][kk]
        centers.append(tmp)

        Rtmp = []
        thresh1 = 1e-6
        thresh2 = 0
        C = 1.0 / (N * nu)
        for ii in range(N):
            if alphas[ii] > thresh1 and C - alphas[ii] > thresh2 and kk == np.argmax(thetas[ii]):
                Rtmp.append(np.linalg.norm(data[ii] - centers[kk]))
        try:
            radii.append(np.min(Rtmp))
        except ValueError:
            tlabels = [np.argmax(thetas[ii]) for ii in range(len(thetas))]
            print kk, Rtmp, Counter(tlabels)
            for ii in range(N):
                if kk == np.argmax(thetas[ii]):
                    print ii, alphas[ii], alphas[ii] > thresh1, C - alphas[ii] > thresh2
            sys.exit(0)

    return centers, radii

def projection_on_probability_simplex(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def update_thetas(data, thetas, alphas, num_epochs=10, lr=1.0):
    # data = np.multiply(alphas, data)
    data = np.matmul(np.diag(alphas), data)
    kernel = np.matmul(data, data.T)

    # print('Theta Loss: {}'.format(compute_theta_loss_2(data, thetas, alphas)))
    tmp_grads = []
    thetas_new = np.zeros(thetas.shape)
    for tt in range(num_epochs):
        print('Theta Iter {} Loss: {}'.format(tt, compute_theta_loss_2(data, thetas, alphas)))
        for ii in range(data.shape[0]):
            tgrad = -1.0 * (kernel[ii][ii] * thetas[ii] + np.matmul(kernel[ii], thetas))
            tnew = thetas[ii] - lr * tgrad
            thetas_new[ii] = projection_on_probability_simplex(tnew)
            tmp_grads.append(tgrad)
        thetas_new, thetas = thetas, thetas_new
        # print('Theta iter {} complete.'.format(tt))
        # np.savetxt('tmp/gradient_{}.csv'.format(tt), thetas)
        # np.savetxt('tmp/thetas_{}.csv'.format(tt), thetas)
    # print('Theta Loss: {}'.format(compute_theta_loss_2(data, thetas, alphas)))
    return thetas

# Hard Assignment for thetas
# We directly use decision function i.e k = min{||f(x) - cen(k)||^2 - Rk^2}
def update_thetas_2(data, thetas, alphas, nu):
    print('Theta Loss Before: {}'.format(compute_theta_loss(data, thetas, alphas)))
    centers, radii = compute_centers_and_radius(data, alphas, thetas, nu)
    thetas_new = np.zeros(thetas.shape)
    for ii in range(data.shape[0]):
        # if alphas[ii] < 1e-6:
        #     thetas_new[ii] = thetas[ii]
        #     continue
        dists = []
        for kk in range(thetas.shape[1]):
            val = np.dot(data[ii] - centers[kk], data[ii] - centers[kk]) - radii[kk] * radii[kk]
            dists.append(val)
        thetas_new[ii][np.argmin(dists)] = 1.0
    # for kk in range(len(radii)):
    #     print kk, radii[kk], centers[kk]
    print('Theta Loss After: {}'.format(compute_theta_loss(data, thetas_new, alphas)))
    return thetas_new

def compute_theta_loss(data, thetas, alphas):
    loss = 0.0
    data_size = len(data)

    for nn in range(data_size):
        # t1 = alphas[nn] * np.dot(data[nn], data[nn])
        t1 = alphas[nn] * np.dot(data[nn], data[nn])
        t2 = 0.0
        for mm in range(data_size):
            t2 += alphas[nn] * alphas[mm] * np.dot(data[nn], data[mm]) * np.dot(thetas[nn], thetas[mm])
        loss += (t1 - t2)
    return loss

def compute_theta_loss_2(data, thetas, alphas):
    loss = 0.0
    data_size = len(data)

    for nn in range(data_size):
        # t1 = alphas[nn] * np.dot(data[nn], data[nn])
        t1 = np.dot(data[nn], data[nn])
        t2 = 0.0
        for mm in range(data_size):
            t2 += np.dot(data[nn], data[mm]) * np.dot(thetas[nn], thetas[mm])
        loss += (t1 - t2)
    return loss

