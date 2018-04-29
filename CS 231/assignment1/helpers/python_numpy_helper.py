l1 = [3, 1, 2]

print(l1[-1]) #prints last element

nums = list(range(5)) #builds a list [0, 1, 2, 3, 4, 5]
print(nums[2:4]) #prints from index 2 to 4 exclusive [2, 3]

animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print("%d: %s" % (idx+1, animal)) #to get index and item

nums = [0,1,2,3,4]
squares = [x ** 2 for x in nums]
even_squares = [x**2 for x in nums if x % 2 == 0]
print(squares)
print(even_squares)

#NUMPY
import numpy as np
a = np.array([1,2,3])
b = np.array([[1,2,3],[4,5,6]])
print(a) #[1 2 3]
print(b) #[[1 2 3] [4 5 6]]
print(b.shape) #(2,3)

z = np.zeros((2,2)) #2x2 array of zeros
o = np.ones((1,2)) #1x2 array of ones
f = np.full((3,3),7) #3x3 array of 7's
i = np.eye(2) #2x2 identity matrix
r1 = np.random.random((2,2)) #2x2 array of random values
r2 = np.random.randn(2,2) #2x2 array of random values from normal distribution
print(r2)

a = np.array([[1,2,3,4], [5,6,7,8],[9,10,11,12]])
b = a[:2,1:3] #get first 2 rows and columns 1 and 2

b[0,0] = 77 #modifies original matrix a
print(a)


a = np.array([[1,2,3,4], [5,6,7,8],[9,10,11,12]])
row_r1 = a[1,:] #gets a rank 1 view of row 1 of a (4,)
row_r2 = a[1:2, :] #gets a rank 2 view of row 1 of a (1,4) in a column vector
print(row_r1.shape)
print(row_r2.shape)

#same for column vectors
a = np.array([[1,2,3,4], [5,6,7,8],[9,10,11,12]])
col_r1 = a[:, 1] #gets rank 1 view of column 1 of a (3,)
col_r2 = a[:, 1:2] #gets rank 2 view of column 1 of a (1, 3) in a row vector

'''array indexing'''
a = np.array([[1,2], [3, 4], [5, 6]])

print(a[[0, 1, 2], [0, 1, 0]]) #is the same as
print(np.array([a[0,0], a[1,1], a[2,0]]))

print(a[[0,0],[1,1]]) #can reuse same elements from array
print(np.array([a[0,1], a[0,1]])) #both print [2,2]

#Useful trick for array indexing

a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
b = np.array([0,2,0,1])
print(a[np.arange(4), b]) #prints([1 6 7 11])
a[np.arange(4),b] += 10

#example inside knn
#sort the row where argsort returns the indexes,
#then get the first k y_train values
#that are minimum in the dists matrix
k = 2
y = np.array([0,1,2,3])
x = np.array([[0,6,3,2],[7,4,5,1],[6,3,1,2]])

for i in range(x.shape[0]):
    closest_y = y[np.argsort(x[i])][:k]
    print(closest_y)
