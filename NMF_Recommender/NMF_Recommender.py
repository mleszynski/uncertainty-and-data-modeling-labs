import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error


class NMFRecommender:

    def __init__(self, random_state=15, tol=1e-3, maxiter=200, rank=3):
        """The parameter values for the algorithm"""
        self.random_state = random_state
        self.tol = tol
        self.maxiter = maxiter
        self.rank = rank

  
    def initialize_matrices(self, m, n):
        """Initialize the W and H matrices"""
        np.random.seed(self.random_state)
        self.W = np.random.rand(m, self.rank)
        self.H = np.random.rand(self.rank, n)
        return self.W, self.H

        
    def compute_loss(self, V, W, H):
        """Computes the loss of the algorithm according to the frobenius norm"""
        return np.linalg.norm((V-W@H))

    
    def update_matrices(self, V, W, H):
        """The multiplicative update step to update W and H"""
        numer = (W.T)@V
        denom = (W.T)@W@H
        H_1 = np.divide(numer, denom)
        H_1 = np.multiply(H, H_1)

        numer = V@(H_1.T)
        denom = W@H_1@(H_1.T)
        W_1 = np.divide(numer, denom)
        W_1 = np.multiply(W, W_1)

        return H_1, W_1

      
    def fit(self, V):
        """Fits W and H weight matrices according to the multiplicative update 
        algorithm. Return W and H"""
        self.initialize_matrices(V.shape[0], V.shape[1])
        for i in range(self.maxiter):
            if self.compute_loss(V, self.W, self.H) < self.tol:
                break
            self.H, self.W = self.update_matrices(V, self.W, self.H)
        return self.W, self.H


    def reconstruct(self):
        """Reconstructs the V matrix for comparison against the original V 
        matrix"""
        return self.W@self.H

        
def prob4():
    """Run NMF recommender on the grocery store example"""
    V = np.array([[0,1,0,1,2,2],
                  [2,3,1,1,2,2],
                  [1,1,1,0,1,1],
                  [0,2,3,4,1,1],
                  [0,0,0,0,1,0]])
    p4_NMF = NMFRecommender(rank=2)
    W, H = p4_NMF.fit(V)
    comp_2_consumers = np.argmax(H, axis=0)
    return W, H, np.sum(comp_2_consumers)


def prob5():
    """Calculate the rank and run NMF
    """
    # read in data #############################################################
    df = pd.read_csv('artist_user.csv')
    data = df.to_numpy()[:,1:]
    benchmark = np.linalg.norm(data)*.0001

    # use sklearn NMF ##########################################################
    for rank in range(3, 14):
        model = NMF(n_components=rank, init='random', random_state=0)
        W = model.fit_transform(data)
        H = model.components_
        V = W@H
        error = np.sqrt(mean_squared_error(data, V))
        if error < benchmark:
            return rank, V
    print('Failed to pass benchmark threshold')


def discover_weekly(user_id=0, p_5=None):
    """
    Create the recommended weekly 30 list for a given user
    """
    # initialize useful dataframes #############################################
    _, V = prob5()
    temp_df = pd.read_csv('artist_user.csv', index_col=0)
    data_array = temp_df.to_numpy()
    mask = data_array == 0  # use a mask to remove already played artists
    V = np.multiply(V, mask)
    names = pd.read_csv('artists.csv', index_col='id')
    rows = temp_df.index.tolist()
    columns = list(temp_df.columns)
    for i in range(len(columns)):  # use artist name instead of id for columns
        columns[i] = names.loc[int(columns[i]), 'name']
    df = pd.DataFrame(data=V, index=rows, columns=columns)
    user_series = df.loc[user_id]

    # find the top 30 ##########################################################
    recommendations = [None]*30
    for i in range(30):
        artist_index = user_series.argmax()
        recommendations[i] = columns[artist_index]
        user_series.iloc[artist_index] = 0

    return recommendations  # returning just a list is fine according to Ethan


# test code ####################################################################
# if __name__ == "__main__":
    # W, H, num_peeps = prob4()
    # print(W)
    # print(H)
    # print(num_peeps)
    # print(prob5())
    # print(discover_weekly(user_id=2))