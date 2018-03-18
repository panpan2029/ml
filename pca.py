# http://sebastianraschka.com/Articles/2014_pca_step_by_step.html

import numpy as np
np.random.seed(2342) # random seed for consistency
# A reader pointed out that Python 2.7 would raise a
# "ValueError: object of too small depth for desired array".
# This can be avoided by choosing a smaller random seed, e.g. 1
# or by completely omitting this line, since I just used the random seed for
# consistency.
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


mu_vec1 = np.array([0,0,0])
cov_mat1 = np.eye(3, dtype=int)
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

mu_vec2 = np.array([1,1,1])
cov_mat2 = np.eye(3, dtype=int)
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
assert class2_sample.shape == (3,20), "The matrix has not the dimensions 3x20"


def paint_test():
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection = '3d')
    plt.rcParams['legend.fontsize'] = 10
    ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
    ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, alpha=0.5, color='red', label='class2')
    plt.title('Samples for class 1 and class 2')
    ax.legend(loc = 'upper right')
    plt.show()


def get_mean_vector():
    all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
    assert all_samples.shape == (3,40), "The matrix has not the dimensions 3x40"
    mean_vector = np.mean(all_samples[0:3, :] , axis = 1)
    # mean_vector.reshape(3,1)
    print mean_vector
    return all_samples, mean_vector

def scatter_matrix(mean_vector, all_samples):
    scatter_matrix = np.zeros((3,3))
    for i in range(all_samples.shape[1]):
        scatter_matrix += (all_samples[:,i].reshape(3,1) - mean_vector).dot((all_samples[:,i].reshape(3,1) - mean_vector).T)
    print('Scatter Matrix:\n', scatter_matrix)
    return scatter_matrix

def cov_matrix(all_samples): #(alternatively to the scatter matrix)
    cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])
    print('Covariance Matrix:\n', cov_mat)
    return cov_mat

def main():
    # paint_test()
    all_samples, mean_vector = get_mean_vector()
    sca_mtr = scatter_matrix(mean_vector, all_samples)
    cov_matrix(all_samples)
    print "test..."

if __name__ == '__main__':
    main()